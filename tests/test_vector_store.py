"""Test per VectorStore.

Uso un FakeEmbedder al posto di Embedder reale: evita di scaricare pesi
di un modello a ogni esecuzione dei test (lento e richiede rete), e rende
i vettori deterministici e facili da ragionare nei test.
"""
from __future__ import annotations

import hashlib

import pytest

from obsidian_brain.chunker import Chunk
from obsidian_brain.vector_store import VectorStore

EMBED_DIM = 16


class FakeEmbedder:
    """Genera vettori deterministici a partire da un hash del testo, cosi'
    che testi identici o simili producano vettori identici/vicini in modo
    prevedibile, senza dover caricare un vero modello.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_text(self, text: str) -> list[float]:
        return self._embed_one(text)

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:EMBED_DIM]]


def make_chunk(content: str, chunk_id: str, note_path="note.md", tags=None, header_path=None) -> Chunk:
    return Chunk(
        id=chunk_id,
        note_title="Nota di test",
        note_path=note_path,
        header_path=header_path or [],
        content=content,
        tags=tags or ["tag1"],
        chunk_index=0,
        total_chunks=1,
    )


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=tmp_path / "chroma_db", embedder=FakeEmbedder())


def test_add_chunks_increases_count(store):
    chunks = [make_chunk("Primo chunk di testo.", "c1"), make_chunk("Secondo chunk di testo.", "c2")]
    store.add_chunks(chunks)
    assert store.count() == 2


def test_add_chunks_with_empty_list_does_nothing(store):
    store.add_chunks([])
    assert store.count() == 0


def test_upsert_overwrites_existing_id_instead_of_duplicating(store):
    store.add_chunks([make_chunk("Versione originale.", "c1")])
    store.add_chunks([make_chunk("Versione aggiornata.", "c1")])
    assert store.count() == 1


def test_query_returns_results_with_expected_shape(store):
    store.add_chunks([make_chunk("Python è un linguaggio di programmazione.", "c1")])
    results = store.query("linguaggio di programmazione", n_results=1)

    assert len(results) == 1
    r = results[0]
    assert set(r.keys()) == {"id", "content", "metadata", "distance"}
    assert r["id"] == "c1"


def test_query_on_empty_store_returns_empty_list(store):
    results = store.query("qualsiasi cosa")
    assert results == []


def test_query_respects_n_results_limit(store):
    chunks = [make_chunk(f"Contenuto numero {i}.", f"c{i}") for i in range(5)]
    store.add_chunks(chunks)
    results = store.query("contenuto", n_results=2)
    assert len(results) == 2


def test_metadata_is_preserved_in_query_results(store):
    chunk = make_chunk(
        "Testo con metadati.",
        "c1",
        note_path="Cartella/Nota.md",
        tags=["a", "b"],
        header_path=["Sezione", "Sottosezione"],
    )
    store.add_chunks([chunk])
    results = store.query("testo", n_results=1)
    meta = results[0]["metadata"]

    assert meta["note_path"] == "Cartella/Nota.md"
    assert meta["tags"] == "a, b"
    assert meta["header_path"] == "Sezione > Sottosezione"


def test_reset_clears_all_data(store):
    store.add_chunks([make_chunk("Da cancellare.", "c1")])
    assert store.count() == 1
    store.reset()
    assert store.count() == 0


def test_data_persists_across_store_instances(tmp_path):
    persist_dir = tmp_path / "chroma_db"

    store1 = VectorStore(persist_dir=persist_dir, embedder=FakeEmbedder())
    store1.add_chunks([make_chunk("Contenuto persistente.", "c1")])
    assert store1.count() == 1

    # nuova istanza, stesso persist_dir: deve ritrovare i dati salvati su disco
    store2 = VectorStore(persist_dir=persist_dir, embedder=FakeEmbedder())
    assert store2.count() == 1