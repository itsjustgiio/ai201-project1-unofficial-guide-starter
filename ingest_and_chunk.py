"""Milestone 3 document ingestion, cleaning, and chunking pipeline.

Loads local project documents, saves raw text, cleans boilerplate/noise, then
creates overlapping word chunks for the retrieval pipeline.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_DIR = Path("documents")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_CHUNK_WORDS = 450
DEFAULT_OVERLAP_WORDS = 65
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf"}
SKIP_FILENAMES = {".gitkeep", "README.md", "readme.md"}


@dataclass
class Document:
    source_id: str
    source_path: str
    title: str
    raw_text: str
    clean_text: str


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    source_path: str
    title: str
    chunk_index: int
    word_start: int
    word_end: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load CCNY source documents, clean text, and produce RAG chunks."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    parser.add_argument("--sample-chunks", type=int, default=5)
    return parser.parse_args()


def discover_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    files: list[Path] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in SKIP_FILENAMES:
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires pdfplumber. Install it with: pip install pdfplumber"
        ) from exc

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(f"\n\n[Page {index}]\n{text}")
    return "\n".join(pages)


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_html(text)
    text = html.unescape(text)
    text = normalize_unicode_spacing(text)

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        if is_boilerplate_line(line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = remove_repeated_blank_lines(text)
    text = remove_repeated_short_lines(text)
    return text.strip()


def strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|section|article)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def normalize_unicode_spacing(text: str) -> str:
    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def is_boilerplate_line(line: str) -> bool:
    lowered = line.lower()
    exact_noise = {
        "advertisement",
        "read more",
        "share",
        "shares",
        "comments",
        "log in",
        "sign up",
        "accept cookies",
        "cookie policy",
        "privacy policy",
        "terms of service",
        "all rights reserved",
    }
    if lowered in exact_noise:
        return True

    noisy_patterns = [
        r"^home\s*(/|$)",
        r"^skip to (main )?content$",
        r"^subscribe\b",
        r"^follow us\b",
        r"^copyright\b",
        r"^©",
        r"^\d+\s+comments?$",
        r"^share (on|this)\b",
        r"^posted by u/",
        r"^view discussions? in",
        r"^back to top$",
    ]
    return any(re.search(pattern, lowered) for pattern in noisy_patterns)


def remove_repeated_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def remove_repeated_short_lines(text: str) -> str:
    lines = text.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if 0 < len(stripped) <= 80:
            counts[stripped.lower()] = counts.get(stripped.lower(), 0) + 1

    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) <= 80 and counts.get(stripped.lower(), 0) >= 4:
            continue
        filtered.append(line)
    return "\n".join(filtered)


def word_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_document(document: Document, chunk_words: int, overlap_words: int) -> list[Chunk]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than 0")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = word_tokens(document.clean_text)
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    step = chunk_words - overlap_words
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_words_list = words[start:end]
        if not chunk_words_list:
            break
        chunk_index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document.source_id}::chunk-{chunk_index:04d}",
                source_id=document.source_id,
                source_path=document.source_path,
                title=document.title,
                chunk_index=chunk_index,
                word_start=start,
                word_end=end,
                text=" ".join(chunk_words_list),
            )
        )
        if end == len(words):
            break
        start += step
    return chunks


def source_id_for(path: Path, input_dir: Path) -> str:
    relative = path.relative_to(input_dir)
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", str(relative.with_suffix("")))
    return safe.strip("-").lower()


def load_documents(input_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in discover_files(input_dir):
        raw_text = load_file(path)
        clean = clean_text(raw_text)
        source_id = source_id_for(path, input_dir)
        documents.append(
            Document(
                source_id=source_id,
                source_path=str(path),
                title=path.stem.replace("_", " ").replace("-", " ").title(),
                raw_text=raw_text,
                clean_text=clean,
            )
        )
    return documents


def write_text_outputs(documents: Iterable[Document], output_dir: Path) -> None:
    raw_dir = output_dir / "raw_text"
    clean_dir = output_dir / "cleaned_text"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    for document in documents:
        (raw_dir / f"{document.source_id}.txt").write_text(
            document.raw_text, encoding="utf-8"
        )
        (clean_dir / f"{document.source_id}.txt").write_text(
            document.clean_text, encoding="utf-8"
        )


def write_chunks(chunks: Iterable[Chunk], output_dir: Path) -> Path:
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / "chunks.jsonl"
    with chunk_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return chunk_path


def write_report(
    documents: list[Document],
    chunks: list[Chunk],
    output_dir: Path,
    chunk_words: int,
    overlap_words: int,
    sample_chunks: list[Chunk],
) -> Path:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "chunk_inspection.md"

    lines = [
        "# Milestone 3 Chunk Inspection",
        "",
        f"- Documents loaded: {len(documents)}",
        f"- Total chunks: {len(chunks)}",
        f"- Chunk size: {chunk_words} words",
        f"- Overlap: {overlap_words} words",
        "",
        "## Document Summary",
        "",
    ]
    for document in documents:
        raw_count = len(word_tokens(document.raw_text))
        clean_count = len(word_tokens(document.clean_text))
        lines.append(
            f"- `{document.source_id}`: {clean_count} cleaned words "
            f"from {raw_count} raw words ({document.source_path})"
        )

    lines.extend(["", "## Representative Chunks", ""])
    for chunk in sample_chunks:
        lines.extend(
            [
                f"### {chunk.chunk_id}",
                "",
                f"Source: `{chunk.source_path}`",
                "",
                chunk.text,
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def pick_representative_chunks(chunks: list[Chunk], sample_count: int) -> list[Chunk]:
    if sample_count <= 0 or not chunks:
        return []
    if len(chunks) <= sample_count:
        return chunks
    positions = [
        round(index * (len(chunks) - 1) / (sample_count - 1))
        for index in range(sample_count)
    ]
    return [chunks[position] for position in positions]


def print_clean_document_sample(documents: list[Document]) -> None:
    document = documents[0]
    preview = document.clean_text[:2500]
    print("\n=== CLEANED DOCUMENT SAMPLE ===")
    print(f"Source: {document.source_path}")
    print(preview)
    if len(document.clean_text) > len(preview):
        print("\n[Sample truncated at 2500 characters]")


def print_chunk_samples(sample_chunks: list[Chunk]) -> None:
    print("\n=== REPRESENTATIVE CHUNKS ===")
    for chunk in sample_chunks:
        print(f"\n--- {chunk.chunk_id} ({chunk.title}) ---")
        print(chunk.text)


def print_count_warning(chunk_count: int, document_count: int) -> None:
    if document_count >= 10 and chunk_count < 50:
        print(
            "\nWARNING: Fewer than 50 chunks across 10+ documents. "
            "Your chunks may be too large for precise retrieval."
        )
    if chunk_count > 2000:
        print(
            "\nWARNING: More than 2,000 chunks. Your chunks may be too small "
            "or your corpus may need source filtering."
        )


def main() -> int:
    args = parse_args()
    documents = load_documents(args.input_dir)

    if not documents:
        print(
            f"No source documents found in {args.input_dir}. Add .txt, .md, "
            ".html, or .pdf files, then run this script again.",
            file=sys.stderr,
        )
        return 1

    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_words=args.chunk_words,
                overlap_words=args.overlap_words,
            )
        )

    write_text_outputs(documents, args.output_dir)
    chunk_path = write_chunks(chunks, args.output_dir)
    sample_chunks = pick_representative_chunks(chunks, args.sample_chunks)
    report_path = write_report(
        documents,
        chunks,
        args.output_dir,
        args.chunk_words,
        args.overlap_words,
        sample_chunks,
    )

    print_clean_document_sample(documents)
    print_chunk_samples(sample_chunks)
    print("\n=== PIPELINE SUMMARY ===")
    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunks written to: {chunk_path}")
    print(f"Inspection report written to: {report_path}")
    print_count_warning(len(chunks), len(documents))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
