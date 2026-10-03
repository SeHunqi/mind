from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from app.core.config import Settings
from app.models.entities import KnowledgeChunk


PRIMARY_RETRIEVAL_LABEL = "Chroma vector + BM25 hybrid + local reranker"
FALLBACK_RETRIEVAL_LABEL = "local BM25 + hybrid_score reranker"


class VectorStoreUnavailable(RuntimeError):
    pass


@dataclass
class VectorSearchHit:
    chunk_id: int | None
    source: str
    source_index: int
    content: str
    score: float


class ChromaKnowledgeStore:
    """Primary RAG path: OpenAI text-embedding-3-small embeddings stored and queried in Chroma."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.can_embed = False
        self.error = ""
        if not settings.knowledge_vector_enabled:
            self.error = "Chroma 向量库未启用"
            return
        if not settings.openai_api_key:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 OPENAI_API_KEY，无法启用 Chroma + text-embedding-3-small 主检索方案")
            self.error = f"缺少 OPENAI_API_KEY，Chroma + text-embedding-3-small 不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return
        try:
            import chromadb
        except ImportError as exc:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 chromadb 依赖，无法启用 Chroma + text-embedding-3-small 主检索方案") from exc
            self.error = f"缺少 chromadb 依赖，Chroma + text-embedding-3-small 不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return

        persist_dir = self._resolve_path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": settings.openai_embedding_model},
        )
        self.can_embed = settings.knowledge_vector_enabled

    def upsert_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        rows = [chunk for chunk in chunks if chunk.id is not None and chunk.content.strip()]
        if not rows:
            return 0
        ids = [self._id(chunk.id) for chunk in rows]
        documents = [chunk.content for chunk in rows]
        metadatas = [
            {"db_id": int(chunk.id), "source": chunk.source, "source_index": int(chunk.source_index)}
            for chunk in rows
        ]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        self.snapshot()
        return len(rows)

    def sync_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        valid_ids = {self._id(int(chunk.id)) for chunk in chunks if chunk.id is not None}
        current_ids = set(self.collection.get().get("ids", []))
        stale_ids = sorted(current_ids - valid_ids)
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return self.upsert_chunks(chunks, embeddings)

    def has_exact_chunk_ids(self, chunks: list[KnowledgeChunk]) -> bool:
        valid_ids = {self._id(int(chunk.id)) for chunk in chunks if chunk.id is not None}
        current_ids = set(self.collection.get().get("ids", []))
        return current_ids == valid_ids

    def delete_source(self, source: str) -> None:
        if not self.can_embed:
            return
        self.collection.delete(where={"source": source})

    def query(self, query_embedding: list[float], top_k: int) -> list[VectorSearchHit]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                VectorSearchHit(
                    chunk_id=int(metadata["db_id"]) if metadata.get("db_id") is not None else None,
                    source=str(metadata.get("source", "")),
                    source_index=int(metadata.get("source_index", 0)),
                    content=document or "",
                    score=1.0 / (1.0 + max(0.0, distance)),
                )
            )
        return hits

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.can_embed:
            raise VectorStoreUnavailable(
                self.error or "Embedding 服务不可用"
            )

        if not texts:
            return []

        batch_size = max(1, self.settings.embedding_batch_size)
        embeddings: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            embeddings.extend(self._embed(batch))

        if len(embeddings) != len(texts):
            raise VectorStoreUnavailable(
                f"Embedding 数量不匹配：输入 {len(texts)} 条，"
                f"返回 {len(embeddings)} 条"
            )

        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1:
            raise VectorStoreUnavailable(
                f"Embedding 向量维度不一致：{sorted(dimensions)}"
            )

        return embeddings

    def snapshot(self) -> str | None:
        if not self.can_embed:
            return None
        if not self.persist_dir.exists():
            return None
        snapshot_root = self._resolve_path(self.settings.chroma_snapshot_dir)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        destination = snapshot_root / datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copytree(self.persist_dir, destination)
        self._prune_snapshots(snapshot_root)
        return str(destination)

    def count(self) -> int:
        if not self.can_embed:
            return 0
        return int(self.collection.count())

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.settings.openai_embedding_model,
            "input": [
                text if text.strip() else " "
                for text in texts
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}"
        }

        url = (
            f"{self.settings.openai_base_url.rstrip('/')}"
            "/embeddings"
        )

        try:
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.settings.embedding_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1000]
            raise VectorStoreUnavailable(
                f"Embedding API 返回错误："
                f"HTTP {exc.response.status_code}，{detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise VectorStoreUnavailable(
                f"Embedding API 请求失败：{type(exc).__name__}: {exc}"
            ) from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise VectorStoreUnavailable(
                "Embedding API 未返回有效 JSON"
            ) from exc

        rows = sorted(
            response_data.get("data", []),
            key=lambda item: item.get("index", 0),
        )
        embeddings = [row.get("embedding") for row in rows]

        if (
                len(embeddings) != len(texts)
                or any(not embedding for embedding in embeddings)
        ):
            raise VectorStoreUnavailable(
                f"Embedding API 返回数量不匹配："
                f"输入 {len(texts)} 条，返回 {len(embeddings)} 条"
            )

        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1:
            raise VectorStoreUnavailable(
                f"Embedding API 返回的向量维度不一致："
                f"{sorted(dimensions)}"
            )

        return [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.settings.project_root / path

    def _prune_snapshots(self, snapshot_root: Path) -> None:
        keep = max(1, self.settings.chroma_snapshot_keep)
        snapshots = sorted([path for path in snapshot_root.iterdir() if path.is_dir()], reverse=True)
        for stale in snapshots[keep:]:
            shutil.rmtree(stale, ignore_errors=True)

    def _id(self, chunk_id: int) -> str:
        return f"knowledge-chunk-{chunk_id}"


ChromaKnowledgeVectorStore = ChromaKnowledgeStore
