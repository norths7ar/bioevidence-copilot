from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from bioevidence.config import Settings, load_settings
from bioevidence.retrieval.scoring import document_text
from bioevidence.schemas.document import Document

EMBEDDING_BATCH_SIZE = 64


class DenseRetrievalError(RuntimeError):
    pass


def create_embedding_client(settings: Settings | None = None) -> OpenAI:
    settings = settings or load_settings()
    if not settings.embedding_api_key:
        raise DenseRetrievalError("BIOEVIDENCE_EMBEDDING_API_KEY is required for dense retrieval")
    if not settings.embedding_base_url:
        raise DenseRetrievalError("BIOEVIDENCE_EMBEDDING_BASE_URL is required for dense retrieval")
    return OpenAI(api_key=settings.embedding_api_key, base_url=settings.embedding_base_url)


def embed_texts(
    texts: Sequence[str],
    *,
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    settings = settings or load_settings()
    if not settings.embedding_model:
        raise DenseRetrievalError("BIOEVIDENCE_EMBEDDING_MODEL is required for dense retrieval")
    if settings.embedding_dimensions is None:
        raise DenseRetrievalError("BIOEVIDENCE_EMBEDDING_DIMENSIONS is required for dense retrieval")
    if settings.embedding_dimensions <= 0:
        raise DenseRetrievalError("BIOEVIDENCE_EMBEDDING_DIMENSIONS must be a positive integer")
    client = client or create_embedding_client(settings)
    embeddings: list[list[float]] = []
    for start_index in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = list(texts[start_index : start_index + EMBEDDING_BATCH_SIZE])
        try:
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
                dimensions=settings.embedding_dimensions,
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
