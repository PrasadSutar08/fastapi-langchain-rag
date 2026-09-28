"""Standalone PDF ingestion utility for the configured raw-data directory.

Normal application uploads should use DocumentService so ownership and document
records are preserved. This utility is intended for local/admin bulk ingestion.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.encoders import jsonable_encoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import logger, settings
from app.services.embedding_service import embedding_service

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR.parent / "data" / "raw"
EXTRACTION_DIR = BASE_DIR.parent / "data" / "extraction"


def load_pdfs(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        logger.info(f"Extracting PDF | file={pdf_path.name}")
        pages = PyPDFLoader(str(pdf_path)).load()
        if not pages:
            continue
        for page in pages:
            page.metadata.update(
                {
                    "filename": pdf_path.name,
                    "source": str(pdf_path),
                }
            )
        documents.extend(pages)
        output = EXTRACTION_DIR / f"{pdf_path.stem}.json"
        output.write_text(
            json.dumps(jsonable_encoder(pages), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return documents


def run() -> None:
    if not RAW_DIR.exists():
        logger.warning(f"PDF input directory does not exist | path={RAW_DIR}")
        return
    EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_pdfs(RAW_DIR)
    if not documents:
        logger.warning(f"No PDFs found | path={RAW_DIR}")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["type"] = "Text"
        chunk.metadata["chunk_index"] = index

    PGVector.from_documents(
        documents=chunks,
        embedding=embedding_service.get_embeddings(),
        collection_name=settings.VECTOR_COLLECTION_NAME,
        connection_string=settings.SYNC_DATABASE_URI,
        pre_delete_collection=False,
    )
    logger.success(f"Bulk ingestion completed | documents={len(documents)} | chunks={len(chunks)}")


if __name__ == "__main__":
    run()
