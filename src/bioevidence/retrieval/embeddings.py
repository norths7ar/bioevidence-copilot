from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from bioevidence.config import Settings, load_settings
from bioevidence.retrieval.scoring import document_text
from bioevidence.schemas.document import Document


DEFAULT_QWEN_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_QWEN_EMBEDDING_DIMENSIONS = 1024
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_BATCH_SIZE = 64


class DenseRetrievalError(RuntimeError):
    pass


def create_embedding_client(settings: Settings | None = None) -> OpenAI:
    settings = settings or load_settings()
    if not settings.qwen_api_key:
        raise DenseRetrievalError("QWEN_API_KEY or DASHSCOPE_API_KEY is required for dense retrieval")
    return OpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url or DEFAULT_QWEN_BASE_URL)


def embed_texts(
    texts: Sequence[str],
    *,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    settings = settings or load_settings()
    client = client or create_embedding_client(settings)
    embeddings: list[list[float]] = []
    for start_index in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = list(texts[start_index : start_index + EMBEDDING_BATCH_SIZE])
        try:
            response = client.embeddings.create(
                model=settings.qwen_embedding_model or DEFAULT_QWEN_EMBEDDING_MODEL,
                input=batch,
                dimensions=settings.qwen_embedding_dimensions or DEFAULT_QWEN_EMBEDDING_DIMENSIONS,
            )
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            raise DenseRetrievalError(f"Embedding request failed: {exc}") from exc
        embeddings.extend([list(item.embedding) for item in response.data])
    if len(embeddings) != len(texts):
        raise DenseRetrievalError("Embedding response length did not match the requested batch")
    return embeddings


def embed_documents(
    documents: Sequence[Document],
    *,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> list[list[float]]:
    return embed_texts([document_text(document) for document in documents], client=client, settings=settings)
