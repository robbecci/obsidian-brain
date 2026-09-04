from __future__ import annotations

import re
from dataclasses import dataclass, field

from obsidian_brain.models import Note

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)


@dataclass
class Chunk:
    id: str
    note_title: str
    note_path: str
    header_path: list[str]
    content: str
    tags: list[str]
    chunk_index: int
    total_chunks: int
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.content)

    def __repr__(self):
        breadcrumb = " > ".join(self.header_path) if self.header_path else "(root)"
        return (
            f"Chunk({self.note_title!r} [{breadcrumb}] "
            f"{self.chunk_index + 1}/{self.total_chunks}, {self.char_count} chars)"
        )


def _split_by_headers(content: str) -> list[tuple[list[str], str]]:
    matches = list(HEADER_PATTERN.finditer(content))
    if not matches:
        return [([], content.strip())] if content.strip() else []

    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []

    preamble = content[: matches[0].start()].strip()
    if preamble:
        sections.append(([], preamble))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_text = content[section_start:section_end].strip()

        header_path = [h for _, h in stack]
        if section_text:
            sections.append((header_path, section_text))

    return sections


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}" if tail else para
        else:
            current = para

        while len(current) > max_chars:
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap:] if overlap else current[max_chars:]

    if current.strip():
        chunks.append(current)

    return chunks


def chunk_note(
    note: Note,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    sections = _split_by_headers(note.content)
    if not sections:
        return []

    raw_chunks: list[tuple[list[str], str]] = []
    for header_path, section_text in sections:
        for piece in _split_text(section_text, max_chars, overlap):
            raw_chunks.append((header_path, piece))

    total = len(raw_chunks)
    chunks = []
    for idx, (header_path, piece) in enumerate(raw_chunks):
        chunks.append(
            Chunk(
                id=f"{note.relative_path}::{idx}",
                note_title=note.title,
                note_path=note.relative_path,
                header_path=header_path,
                content=piece,
                tags=note.tags,
                chunk_index=idx,
                total_chunks=total,
            )
        )
    return chunks


def chunk_notes(
    notes: list[Note],
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    all_chunks = []
    for note in notes:
        all_chunks.extend(chunk_note(note, max_chars=max_chars, overlap=overlap))
    return all_chunks