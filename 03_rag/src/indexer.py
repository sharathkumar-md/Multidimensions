from __future__ import annotations

import json
import uuid
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient, models

from config.settings import settings
from src.chunker import Chunk


_COLLECTION = settings.qdrant_collection_name
_CHUNKS_FILE = "chunks.json"

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"

_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"


def _qdrant_client(index_dir: Path) -> QdrantClient:
    return QdrantClient(
        path=str(index_dir / "qdrant_db"),
        local_inference_batch_size=8,
    )


def build_index(
    chunks: list[Chunk],
    index_dir: Path | None = None,
) -> None:
    index_dir = Path(index_dir or settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"building index: {len(chunks)} chunks → {index_dir}"
    )

    client = _qdrant_client(index_dir)

    try:
        # ---------------------------------------------------------
        # Determine embedding dimension from the configured model.
        # This avoids hard-coding BGE-large's dimension.
        # ---------------------------------------------------------

        dense_size = client.get_embedding_size(
            settings.embed_model
        )

        logger.info(
            f"dense model: {settings.embed_model} "
            f"(dimension={dense_size})"
        )

        # ---------------------------------------------------------
        # Recreate collection
        # ---------------------------------------------------------

        try:
            client.delete_collection(
                collection_name=_COLLECTION
            )
        except Exception as exc:
            logger.debug(
                f"Collection '{_COLLECTION}' did not exist yet: {exc}"
            )

        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config={
                _DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

        # ---------------------------------------------------------
        # Prepare documents
        # ---------------------------------------------------------

        texts = [chunk.text for chunk in chunks]

        ids = [
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                chunk.chunk_id,
            ).hex
            for chunk in chunks
        ]

        payloads = [
            {
                "source_doc": chunk.source_doc,
                "page_num": chunk.page_num,
                "doc_hash": chunk.doc_hash,
            }
            for chunk in chunks
        ]

        # ---------------------------------------------------------
        # Dense + sparse vectors
        #
        # Qdrant's current FastEmbed integration performs the
        # embedding when models.Document / models.SparseTextEmbedding
        # are supplied.
        # ---------------------------------------------------------

        logger.info(
            f"embedding {len(texts)} documents..."
        )

        dense_documents = [
            models.Document(
                text=text,
                model=settings.embed_model,
            )
            for text in texts
        ]

        # Generate sparse embeddings explicitly because the current
        # qdrant-client no longer exposes the old set_sparse_model()
        # helper.
        from fastembed import SparseTextEmbedding

        sparse_model = SparseTextEmbedding(
            model_name=_SPARSE_MODEL
        )

        sparse_embeddings = list(
            sparse_model.embed(
                texts,
                batch_size=64,
            )
        )

        # ---------------------------------------------------------
        # Upload points
        # ---------------------------------------------------------

        points = []

        for point_id, dense_doc, sparse_embedding, payload in zip(
            ids,
            dense_documents,
            sparse_embeddings,
            payloads,
        ):
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        _DENSE_VECTOR_NAME: dense_doc,
                        _SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_embedding.indices.tolist(),
                            values=sparse_embedding.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )

        batch_size = 64

        for start in range(
            0,
            len(points),
            batch_size,
        ):
            batch = points[start : start + batch_size]

            client.upload_points(
                collection_name=_COLLECTION,
                points=batch,
            )

            logger.debug(
                f"indexed "
                f"{min(start + batch_size, len(points))}/{len(points)}"
            )

        # ---------------------------------------------------------
        # Persist chunk metadata
        # ---------------------------------------------------------

        with open(
            index_dir / _CHUNKS_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_hash": chunk.doc_hash,
                        "source_doc": chunk.source_doc,
                        "page_num": chunk.page_num,
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("index done")

    finally:
        try:
            client.close()
        except Exception as exc:
            logger.warning(
                f"Failed to close Qdrant client: {exc}"
            )


def load_index(
    index_dir: Path | None = None,
) -> tuple[QdrantClient, list[Chunk]]:
    index_dir = Path(index_dir or settings.index_dir)

    client = _qdrant_client(index_dir)

    chunks_file = index_dir / _CHUNKS_FILE

    if not chunks_file.exists():
        logger.warning(
            f"No chunks file found at {chunks_file}. "
            "Returning empty index."
        )
        return client, []

    with open(
        chunks_file,
        encoding="utf-8",
    ) as f:
        chunk_dicts = json.load(f)

    chunks = [
        Chunk(
            chunk_id=data["chunk_id"],
            doc_hash=data["doc_hash"],
            source_doc=data["source_doc"],
            page_num=data["page_num"],
            text=data["text"],
        )
        for data in chunk_dicts
    ]

    logger.info(
        f"index loaded: {len(chunks)} chunks"
    )

    return client, chunks