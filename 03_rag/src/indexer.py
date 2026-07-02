from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient

from config.settings import settings
from src.chunker import Chunk

_COLLECTION = "rag_chunks"
_CHUNKS_FILE = "chunks.json"


def _qdrant_client(index_dir: Path) -> QdrantClient:
    return QdrantClient(path=str(index_dir / "qdrant_db"))


def build_index(chunks: list[Chunk], index_dir: Path | None = None) -> None:
    index_dir = Path(index_dir or settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"building index: {len(chunks)} chunks → {index_dir}")

    client = _qdrant_client(index_dir)
    
    # Configure fastembed models
    client.set_model(settings.embed_model)
    client.set_sparse_model("prithivida/Splade_PP_en_v1")

    # Delete then recreate — recreate_collection was removed in qdrant-client ≥1.9
    try:
        client.delete_collection(collection_name=_COLLECTION)
    except Exception:
        pass  # collection may not exist on first build
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config=client.get_fastembed_vector_params(),
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
    )

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {"source_doc": c.source_doc, "page_num": c.page_num, "doc_hash": c.doc_hash}
        for c in chunks
    ]

    batch = 64
    for i in range(0, len(chunks), batch):
        client.add(
            collection_name=_COLLECTION,
            documents=texts[i : i + batch],
            ids=ids[i : i + batch],
            metadata=metadatas[i : i + batch],
        )
        logger.debug(f"indexed {min(i + batch, len(chunks))}/{len(chunks)}")

    with open(index_dir / _CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            [{"chunk_id": c.chunk_id, "doc_hash": c.doc_hash, "source_doc": c.source_doc,
              "page_num": c.page_num, "text": c.text} for c in chunks],
            f, ensure_ascii=False,
        )

    logger.info("index done")


def load_index(index_dir: Path | None = None) -> tuple[QdrantClient, list[Chunk]]:
    index_dir = Path(index_dir or settings.index_dir)

    client = _qdrant_client(index_dir)
    # Re-declare models on load so client knows how to embed queries
    client.set_model(settings.embed_model)
    client.set_sparse_model("prithivida/Splade_PP_en_v1")

    with open(index_dir / _CHUNKS_FILE, encoding="utf-8") as f:
        chunk_dicts = json.load(f)

    chunks = [
        Chunk(chunk_id=d["chunk_id"], doc_hash=d["doc_hash"], source_doc=d["source_doc"],
              page_num=d["page_num"], text=d["text"])
        for d in chunk_dicts
    ]

    logger.info(f"index loaded: {len(chunks)} chunks")
    return client, chunks
