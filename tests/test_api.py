"""Test per l'API FastAPI.

Uso create_app() con un Responder costruito su un VectorStore con
embedder finto e un FakeChatClient (stessa tecnica di test_responder.py),
cosi' da testare gli endpoint HTTP end-to-end senza indicizzare una vault
vera ne' avere Ollama in esecuzione.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from obsidian_brain.api import create_app
from obsidian_brain.chunker import Chunk
from obsidian_brain.responder import Responder
from obsidian_brain.retriever import Retriever
from obsidian_brain.vector_store import VectorStore


class ScriptedEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[t] for t in texts]

    def embed_text(self, text: str) -> list[float]:
        return self.vectors[text]


class FakeChatClient:
    def __init__(self, response_text: str = "Risposta finta."):
        self.response_text = response_text

    def chat(self, model: str, messages: list[dict]) -> str:
        return self.response_text


def make_chunk(content: str, chunk_id: str, note_title: str = "Nota") -> Chunk:
    return Chunk(
        id=chunk_id,
        note_title=note_title,
        note_path=f"{note_title}.md",
        header_path=[],
        content=content,
        tags=[],
        chunk_index=0,
        total_chunks=1,
    )


def make_responder_with_hit(tmp_path, response_text: str = "Risposta finta.") -> Responder:
    vectors = {"domanda": [1.0, 0.0], "contenuto pertinente": [1.0, 0.0]}
    embedder = ScriptedEmbedder(vectors)
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)
    store.add_chunks([make_chunk("contenuto pertinente", "c1", note_title="Python")])
    return Responder(Retriever(store), client=FakeChatClient(response_text))


def make_responder_without_hit(tmp_path) -> Responder:
    embedder = ScriptedEmbedder({"domanda": [1.0, 0.0]})
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)
    return Responder(Retriever(store), client=FakeChatClient())


def test_health_endpoint_reports_indexed_chunk_count(tmp_path):
    responder = make_responder_with_hit(tmp_path)
    client = TestClient(create_app(responder))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] == 1


def test_ask_returns_answer_with_sources_when_relevant(tmp_path):
    responder = make_responder_with_hit(tmp_path, response_text="Python è un linguaggio.")
    client = TestClient(create_app(responder))

    response = client.post("/ask", json={"question": "domanda"})

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["answer"] == "Python è un linguaggio."
    assert body["sources"] == ["Python"]


def test_ask_returns_not_found_when_no_relevant_chunk(tmp_path):
    responder = make_responder_without_hit(tmp_path)
    client = TestClient(create_app(responder))

    response = client.post("/ask", json={"question": "domanda"})

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is False
    assert body["sources"] == []


def test_ask_rejects_request_without_question_field(tmp_path):
    responder = make_responder_with_hit(tmp_path)
    client = TestClient(create_app(responder))

    response = client.post("/ask", json={})

    assert response.status_code == 422  # errore di validazione Pydantic


def test_ask_rejects_empty_question(tmp_path):
    responder = make_responder_with_hit(tmp_path)
    client = TestClient(create_app(responder))

    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422