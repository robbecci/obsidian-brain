"""Test per Retriever.

Uso uno ScriptedEmbedder che mappa ogni testo a un vettore scelto a mano,
cosi' da controllare ESATTAMENTE la distanza coseno tra query e chunk nei
test (es. vettori identici = distanza 0, ortogonali = distanza 1, opposti
= distanza 2), invece di dipendere da un modello reale il cui output non
e' facilmente prevedibile.
"""
from __future__ import annotations

import pytest

from obsidian_brain.chunker import Chunk
from obsidian_brain.retriever import Retriever, DEFAULT_MAX_DISTANCE
from obsidian_brain.vector_store import VectorStore


class ScriptedEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[t] for t in texts]

    def embed_text(self, text: str) -> list[float]:
        return self.vectors[text]


def make_chunk(content: str, chunk_id: str, note_title="Nota", header_path=None) -> Chunk:
    return Chunk(
        id=chunk_id,
        note_title=note_title,
        note_path=f"{note_title}.md",
        header_path=header_path or [],
        content=content,
        tags=["tag"],
        chunk_index=0,
        total_chunks=1,
    )


def test_retrieve_found_true_when_chunk_is_identical_to_query(tmp_path):
    vectors = {"domanda": [1.0, 0.0], "chunk vicino": [1.0, 0.0]}
    embedder = ScriptedEmbedder(vectors)
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)
    store.add_chunks([make_chunk("chunk vicino", "c1")])

    retriever = Retriever(store)
    result = retriever.retrieve("domanda")

    assert result.found is True
    assert len(result.chunks) == 1


def test_retrieve_found_false_when_only_orthogonal_chunk_exists(tmp_path):
    # vettori ortogonali -> distanza coseno = 1, ben oltre la soglia di default 0.6
    vectors = {"domanda": [1.0, 0.0], "chunk lontano": [0.0, 1.0]}
    embedder = ScriptedEmbedder(vectors)
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)
    store.add_chunks([make_chunk("chunk lontano", "c1")])

    retriever = Retriever(store)
    result = retriever.retrieve("domanda")

    assert result.found is False
    assert result.chunks == []


def test_retrieve_filters_out_irrelevant_chunks_keeping_relevant_ones(tmp_path):
    vectors = {
        "domanda": [1.0, 0.0],
        "chunk vicino": [1.0, 0.0],   # distanza 0 -> incluso
        "chunk lontano": [0.0, 1.0],  # distanza 1 -> escluso
    }
    embedder = ScriptedEmbedder(vectors)
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)
    store.add_chunks([
        make_chunk("chunk vicino", "c1"),
        make_chunk("chunk lontano", "c2"),
    ])

    retriever = Retriever(store, n_results=2)
    result = retriever.retrieve("domanda")

    assert result.found is True
    ids = [c["id"] for c in result.chunks]
    assert ids == ["c1"]


def test_stricter_threshold_can_exclude_a_previously_included_chunk(tmp_path):
    # distanza coseno tra [1,0] e [0.8, 0.6] normalizzato: circa 0.2
    vectors = {"domanda": [1.0, 0.0], "chunk medio": [0.8, 0.6]}
    embedder = ScriptedEmbedder(vectors)
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)
    store.add_chunks([make_chunk("chunk medio", "c1")])

    permissive = Retriever(store, max_distance=0.5)
    strict = Retriever(store, max_distance=0.05)

    assert permissive.retrieve("domanda").found is True
    assert strict.retrieve("domanda").found is False


def test_retrieve_on_empty_store_returns_not_found(tmp_path):
    embedder = ScriptedEmbedder({"domanda": [1.0, 0.0]})
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)

    retriever = Retriever(store)
    result = retriever.retrieve("domanda")

    assert result.found is False
    assert result.chunks == []


def test_as_context_includes_source_and_content(tmp_path):
    vectors = {"domanda": [1.0, 0.0], "Il contenuto del chunk.": [1.0, 0.0]}
    embedder = ScriptedEmbedder(vectors)
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)
    store.add_chunks([
        make_chunk("Il contenuto del chunk.", "c1", note_title="Python", header_path=["Setup"])
    ])

    retriever = Retriever(store)
    result = retriever.retrieve("domanda")
    context = result.as_context()

    assert "Python" in context
    assert "Setup" in context
    assert "Il contenuto del chunk." in context


def test_as_context_on_empty_result_returns_empty_string(tmp_path):
    embedder = ScriptedEmbedder({"domanda": [1.0, 0.0]})
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)

    retriever = Retriever(store)
    result = retriever.retrieve("domanda")

    assert result.as_context() == ""


def test_default_max_distance_is_within_valid_cosine_range():
    assert 0 < DEFAULT_MAX_DISTANCE < 2