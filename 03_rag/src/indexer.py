from __future__ import annotations

import json
import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
from loguru import logger

from config.settings import settings
from src.chunker import Chunk
from src.embed import embed_texts

_CHROMA_COLLECTION = "rag_chunks"
_BM25_FILE = "bm25.pkl"
_CHUNKS_FILE = "chunks.json"


def _chroma_client(index_dir: Path) -> chromadb.Client:
    return chromadb.Client(
        ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(index_dir),
            anonymized_telemetry=False,
        )
    )


def build_index(chunks: list[Chunk], index_dir: Path | None = None) -> None:
    index_dir = Path(index_dir or settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building index with {len(chunks)} chunks → {index_dir}")

    # --- ChromaDB dense index ---
    client = _chroma_client(index_dir)
    try:
        client.delete_collection(_CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=_CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {"source_doc": c.source_doc, "page_num": c.page_num, "doc_hash": c.doc_hash}
        for c in chunks
    ]

    batch = 256
    for i in range(0, len(chunks), batch):
        batch_texts = texts[i : i + batch]
        embeddings = embed_texts(batch_texts)
        collection.add(
            ids=ids[i : i + batch],
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=metadatas[i : i + batch],
        )
        logger.debug(f"Indexed batch {i // batch + 1}/{(len(chunks) + batch - 1) // batch}")

    client.persist()
    logger.info("ChromaDB index built and persisted")

    # --- BM25 sparse index ---
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(index_dir / _BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    logger.info("BM25 index saved")

    # --- Chunk metadata store ---
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "doc_hash": c.doc_hash,
            "source_doc": c.source_doc,
            "page_num": c.page_num,
            "text": c.text,
        }
        for c in chunks
    ]
    with open(index_dir / _CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False)
    logger.info(f"Index complete: {len(chunks)} chunks stored")


def load_index(index_dir: Path | None = None) -> tuple[chromadb.Collection, BM25Okapi, list[Chunk]]:
    index_dir = Path(index_dir or settings.index_dir)
    logger.info(f"Loading index from {index_dir}")

    client = _chroma_client(index_dir)
    collection = client.get_collection(_CHROMA_COLLECTION)

    with open(index_dir / _BM25_FILE, "rb") as f:
        bm25: BM25Okapi = pickle.load(f)

    with open(index_dir / _CHUNKS_FILE, encoding="utf-8") as f:
        chunk_dicts = json.load(f)

    chunks = [
        Chunk(
            chunk_id=d["chunk_id"],
            doc_hash=d["doc_hash"],
            source_doc=d["source_doc"],
            page_num=d["page_num"],
            text=d["text"],
        )
        for d in chunk_dicts
    ]

    logger.info(f"Index loaded: {len(chunks)} chunks")
    return collection, bm25, chunks
