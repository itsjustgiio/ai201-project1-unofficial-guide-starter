"""Milestone 5 grounded generation layer.

Retrieves relevant chunks, asks Groq to answer using only those chunks, and
returns programmatic source attribution for the interface.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from embed_and_retrieve import (
    DEFAULT_COLLECTION,
    DEFAULT_DB_DIR,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    RetrievedChunk,
    retrieve,
)


DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MIN_CONFIDENT_DISTANCE = 0.5

SYSTEM_PROMPT = """You are a grounded question-answering assistant for a CCNY unofficial guide.

You must follow these rules:
1. Answer using only the provided retrieved document chunks.
2. Do not use outside knowledge, even if you think you know the answer.
3. If the chunks do not contain enough information, say exactly: "I don't have enough information on that."
4. Do not invent professor names, course details, policies, dates, or student opinions.
5. Keep the answer concise and specific to the retrieved evidence.
"""


@dataclass
class AskResult:
    answer: str
    sources: list[str]
    chunks: list[dict[str, Any]]


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for chunk in chunks:
        title = chunk.metadata.get("title", "Unknown source")
        source_path = chunk.metadata.get("source_path", "Unknown path")
        chunk_index = chunk.metadata.get("chunk_index", "?")
        blocks.append(
            "\n".join(
                [
                    f"[Source {chunk.rank}]",
                    f"Title: {title}",
                    f"File: {source_path}",
                    f"Chunk: {chunk_index}",
                    f"Distance: {chunk.distance:.4f}",
                    "Text:",
                    chunk.text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def source_label(chunk: RetrievedChunk) -> str:
    title = chunk.metadata.get("title", "Unknown source")
    source_path = chunk.metadata.get("source_path", "Unknown path")
    chunk_index = chunk.metadata.get("chunk_index", "?")
    return f"{title} ({source_path}, chunk {chunk_index}, distance {chunk.distance:.4f})"


def unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        label = source_label(chunk)
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


def chunk_payload(chunk: RetrievedChunk) -> dict[str, Any]:
    payload = asdict(chunk)
    payload["source"] = source_label(chunk)
    return payload


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"""Question:
{question}

Retrieved document chunks:
{format_context(chunks)}

Answer the question using only the retrieved document chunks above. If the answer is not clearly supported by those chunks, say exactly: "I don't have enough information on that."
"""


def get_groq_client() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and set your Groq key."
        )
    return Groq(api_key=api_key)


def generate_answer(question: str, chunks: list[RetrievedChunk], model_name: str) -> str:
    client = get_groq_client()
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, chunks)},
        ],
    )
    answer = completion.choices[0].message.content
    return (answer or "").strip()


def ask(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    db_dir: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    min_confident_distance: float = DEFAULT_MIN_CONFIDENT_DISTANCE,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "chunks": [],
        }

    chunks = retrieve(
        query=question,
        db_dir=db_dir,
        collection_name=collection_name,
        model_name=embedding_model,
        top_k=top_k,
    )

    if not chunks:
        return {
            "answer": "I don't have enough information on that.",
            "sources": [],
            "chunks": [],
        }

    best_distance = chunks[0].distance
    if best_distance > min_confident_distance:
        return {
            "answer": "I don't have enough information on that.",
            "sources": unique_sources(chunks),
            "chunks": [chunk_payload(chunk) for chunk in chunks],
        }

    answer = generate_answer(question, chunks, model_name=llm_model)
    return {
        "answer": answer,
        "sources": unique_sources(chunks),
        "chunks": [chunk_payload(chunk) for chunk in chunks],
    }


def print_result(result: dict[str, Any]) -> None:
    print("\nANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    if result["sources"]:
        for source in result["sources"]:
            print(f"- {source}")
    else:
        print("- No sources retrieved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the grounded CCNY RAG system.")
    parser.add_argument("question", help="Question to answer from retrieved documents.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = ask(
            question=args.question,
            top_k=args.top_k,
            llm_model=args.llm_model,
        )
        print_result(result)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
