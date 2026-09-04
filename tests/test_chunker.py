from pathlib import Path

import pytest

from obsidian_brain.chunker import chunk_note, chunk_notes, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from obsidian_brain.models import Note


def make_note(content: str, title="Test Note", relative_path="test.md", tags=None) -> Note:
    return Note(
        path=Path(f"/fake/{relative_path}"),
        relative_path=relative_path,
        title=title,
        content=content,
        tags=tags or ["tag1", "tag2"],
    )


def test_short_note_produces_single_chunk():
    note = make_note("Testo breve, sotto la soglia di chunking.")
    chunks = chunk_note(note)
    assert len(chunks) == 1
    assert chunks[0].content == note.content
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1


def test_metadata_propagates_to_every_chunk():
    long_paragraph = " ".join(["parola"] * 400)
    content = "\n\n".join([long_paragraph] * 3)
    note = make_note(content, title="Nota Lunga", relative_path="Cartella/Lunga.md", tags=["x", "y"])
    chunks = chunk_note(note)

    assert len(chunks) > 1
    for c in chunks:
        assert c.note_title == "Nota Lunga"
        assert c.note_path == "Cartella/Lunga.md"
        assert c.tags == ["x", "y"]


def test_chunk_ids_are_unique_and_sequential():
    long_paragraph = " ".join(["parola"] * 400)
    content = "\n\n".join([long_paragraph] * 3)
    note = make_note(content, relative_path="doc.md")
    chunks = chunk_note(note)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [f"doc.md::{i}" for i in range(len(chunks))]


def test_header_path_is_captured():
    content = (
        "Testo introduttivo senza header.\n\n"
        "## Setup\n\n"
        "Contenuto della sezione Setup.\n\n"
        "### Installazione\n\n"
        "Contenuto della sottosezione Installazione.\n\n"
        "## Utilizzo\n\n"
        "Contenuto della sezione Utilizzo."
    )
    note = make_note(content)
    chunks = chunk_note(note)

    header_paths = [c.header_path for c in chunks]
    assert [] in header_paths
    assert ["Setup"] in header_paths
    assert ["Setup", "Installazione"] in header_paths
    assert ["Utilizzo"] in header_paths


def test_header_stack_resets_correctly_after_subsection():
    content = (
        "## A\n\ncontenuto A\n\n"
        "### A.1\n\ncontenuto A.1\n\n"
        "## B\n\ncontenuto B"
    )
    note = make_note(content)
    chunks = chunk_note(note)
    header_paths = [c.header_path for c in chunks]
    assert ["A"] in header_paths
    assert ["A", "A.1"] in header_paths
    assert ["B"] in header_paths
    assert ["A", "A.1", "B"] not in header_paths


def test_long_text_is_split_into_multiple_chunks_within_size_limit():
    long_paragraph = " ".join(["parola"] * 400)
    content = "\n\n".join([long_paragraph] * 5)
    note = make_note(content)
    chunks = chunk_note(note, max_chars=500, overlap=50)

    assert len(chunks) > 1
    for c in chunks:
        assert c.char_count <= 500 + 60


def test_overlap_creates_shared_content_between_consecutive_chunks():
    long_paragraph = " ".join(["parola"] * 400)
    content = "\n\n".join([long_paragraph] * 3)
    note = make_note(content)
    chunks = chunk_note(note, max_chars=500, overlap=100)

    assert len(chunks) > 1
    tail_of_first = chunks[0].content[-100:]
    assert tail_of_first[:20] in chunks[1].content


def test_chunk_notes_flattens_multiple_notes():
    note1 = make_note("Nota corta 1.", relative_path="n1.md")
    note2 = make_note("Nota corta 2.", relative_path="n2.md")
    chunks = chunk_notes([note1, note2])
    assert len(chunks) == 2
    assert {c.note_path for c in chunks} == {"n1.md", "n2.md"}


def test_empty_note_produces_no_chunks():
    note = make_note("   \n\n   ")
    chunks = chunk_note(note)
    assert chunks == []


def test_default_constants_are_reasonable():
    assert DEFAULT_CHUNK_SIZE > DEFAULT_CHUNK_OVERLAP
    assert DEFAULT_CHUNK_OVERLAP > 0