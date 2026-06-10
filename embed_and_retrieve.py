"""Milestone 4 embedding and retrieval pipeline.

Builds a ChromaDB vector store from the chunks produced by ingest_and_chunk.py,
then retrieves the top-k most relevant chunks for test queries.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
DEFAULT_DB_DIR = Path("chroma_db")
DEFAULT_COLLECTION = "ccny_unofficial_guide_chunks"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

EVALUATION_QUERIES = [
    "What advice do current students give incoming CCNY CS freshmen?",
    "What complaints do students have about registration?",
    "What traits do students praise in highly rated professors?",
]


@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    rank: int
    chunk_id: str
    text: str
    distance: float
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunks with sentence-transformers and test ChromaDB retrieval."
    )
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("index", help="Embed chunks and write/update the ChromaDB index.")

    query_parser = subparsers.add_parser("query", help="Retrieve chunks for one query.")
    query_parser.add_argument("query", help="Question to search for.")

    subparsers.add_parser(
        "test",
        help="Run the first three planning.md evaluation queries and print retrieval results.",
    )
    return parser.parse_args()


def import_dependencies():
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Milestone 4 dependencies are missing. Install them with: "
            "pip install -r requirements.txt"
        ) from exc
    return chromadb, SentenceTransformer


def load_chunks(chunks_path: Path) -> list[StoredChunk]:
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunks_path}. Run `python ingest_and_chunk.py` first."
        )

    chunks: list[StoredChunk] = []
    with chunks_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {chunks_path} on line {line_number}: {exc}"
                ) from exc

            required = {"chunk_id", "text", "source_path", "title", "chunk_index"}
            missing = required - set(record)
            if missing:
                raise ValueError(
                    f"Chunk record on line {line_number} is missing: {sorted(missing)}"
                )

            metadata = {
                "source_id": str(record.get("source_id", "")),
                "source_path": str(record["source_path"]),
                "title": str(record["title"]),
                "chunk_index": int(record["chunk_index"]),
                "word_start": int(record.get("word_start", 0)),
                "word_end": int(record.get("word_end", 0)),
            }
            chunks.append(
                StoredChunk(
                    chunk_id=str(record["chunk_id"]),
                    text=str(record["text"]),
                    metadata=metadata,
                )
            )

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}.")
    return chunks


def get_collection(db_dir: Path, collection_name: str):
    chromadb, _ = import_dependencies()
    client = chromadb.PersistentClient(path=str(db_dir))
    return client.get_or_create_collection(name=collection_name)


def load_model(model_name: str):
    _, SentenceTransformer = import_dependencies()
    return SentenceTransformer(model_name)


def build_index(
    chunks_path: Path,
    db_dir: Path,
    collection_name: str,
    model_name: str,
) -> int:
    chunks = load_chunks(chunks_path)
    model = load_model(model_name)
    collection = get_collection(db_dir, collection_name)

    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=texts,
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=embeddings,
    )
    return len(chunks)


def retrieve(
    query: str,
    db_dir: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    model = load_model(model_name)
    collection = get_collection(db_dir, collection_name)

    if collection.count() == 0:
        raise ValueError(
            "The ChromaDB collection is empty. Run `python embed_and_retrieve.py index` first."
        )

    query_embedding = model.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[RetrievedChunk] = []
    for index, chunk_id in enumerate(ids):
        retrieved.append(
            RetrievedChunk(
                rank=index + 1,
                chunk_id=chunk_id,
                text=documents[index],
                distance=float(distances[index]),
                metadata=dict(metadatas[index]),
            )
        )
    return retrieved


def print_results(query: str, results: list[RetrievedChunk]) -> None:
    print(f"\nQUERY: {query}")
    for result in results:
        title = result.metadata.get("title", "Unknown source")
        source = result.metadata.get("source_path", "Unknown path")
        chunk_index = result.metadata.get("chunk_index", "?")
        print(
            f"\n[{result.rank}] distance={result.distance:.4f} "
            f"source={title} chunk={chunk_index}"
        )
        print(f"    file: {source}")
        print(result.text)


def run_test_queries(
    db_dir: Path,
    collection_name: str,
    model_name: str,
    top_k: int,
) -> None:
    for query in EVALUATION_QUERIES:
        results = retrieve(
            query=query,
            db_dir=db_dir,
            collection_name=collection_name,
            model_name=model_name,
            top_k=top_k,
        )
        print_results(query, results)


def main() -> int:
    args = parse_args()
    command = args.command or "test"

    try:
        if command == "index":
            count = build_index(
                chunks_path=args.chunks_path,
                db_dir=args.db_dir,
                collection_name=args.collection,
                model_name=args.model,
            )
            print(
                f"Indexed {count} chunks into ChromaDB collection "
                f"`{args.collection}` at {args.db_dir}."
            )
            return 0

        if command == "query":
            results = retrieve(
                query=args.query,
                db_dir=args.db_dir,
                collection_name=args.collection,
                model_name=args.model,
                top_k=args.top_k,
            )
            print_results(args.query, results)
            return 0

        if command == "test":
            run_test_queries(
                db_dir=args.db_dir,
                collection_name=args.collection,
                model_name=args.model,
                top_k=args.top_k,
            )
            return 0

        raise ValueError(f"Unknown command: {command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
