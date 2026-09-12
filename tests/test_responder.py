"""Test per Responder.

Uso un FakeChatClient che imita l'interfaccia ChatClient (chat(model,
messages) -> str), senza bisogno di avere Ollama installato o in
esecuzione - permette di testare la logica di grounding (quando chiamare
il modello, cosa includere nel prompt, come processare la risposta)
in isolamento.
"""
from __future__ import annotations

from obsidian_brain.chunker import Chunk
from obsidian_brain.responder import Responder, NOT_FOUND_MESSAGE
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
        self.calls: list[dict] = []

    def chat(self, model: str, messages: list[dict]) -> str:
        self.calls.append({"model": model, "messages": messages})
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


def make_retriever_with_hit(tmp_path):
    vectors = {"domanda": [1.0, 0.0], "contenuto pertinente": [1.0, 0.0]}
    embedder = ScriptedEmbedder(vectors)
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)
    store.add_chunks([make_chunk("contenuto pertinente", "c1", note_title="Python")])
    return Retriever(store)


def make_retriever_without_hit(tmp_path):
    embedder = ScriptedEmbedder({"domanda": [1.0, 0.0]})
    store = VectorStore(persist_dir=tmp_path / "db", embedder=embedder)
    return Retriever(store)


def test_answer_returns_not_found_without_calling_model(tmp_path):
    retriever = make_retriever_without_hit(tmp_path)
    client = FakeChatClient()
    responder = Responder(retriever, client=client)

    result = responder.answer("domanda")

    assert result.found is False
    assert result.text == NOT_FOUND_MESSAGE
    assert result.sources == []
    assert client.calls == []  # il modello non deve essere chiamato


def test_answer_calls_model_when_relevant_context_found(tmp_path):
    retriever = make_retriever_with_hit(tmp_path)
    client = FakeChatClient(response_text="Python è un linguaggio.")
    responder = Responder(retriever, client=client)

    result = responder.answer("domanda")

    assert result.found is True
    assert result.text == "Python è un linguaggio."
    assert len(client.calls) == 1


def test_prompt_includes_context_and_query(tmp_path):
    retriever = make_retriever_with_hit(tmp_path)
    client = FakeChatClient()
    responder = Responder(retriever, client=client)

    responder.answer("domanda")

    messages = client.calls[0]["messages"]
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert "contenuto pertinente" in user_message
    assert "domanda" in user_message


def test_system_prompt_is_sent_as_system_message(tmp_path):
    retriever = make_retriever_with_hit(tmp_path)
    client = FakeChatClient()
    responder = Responder(retriever, client=client)

    responder.answer("domanda")

    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert len(messages[0]["content"]) > 0


def test_sources_reflect_note_titles_of_retrieved_chunks(tmp_path):
    retriever = make_retriever_with_hit(tmp_path)
    client = FakeChatClient()
    responder = Responder(retriever, client=client)

    result = responder.answer("domanda")

    assert result.sources == ["Python"]


def test_model_default_can_be_overridden(tmp_path):
    retriever = make_retriever_with_hit(tmp_path)
    client = FakeChatClient()
    responder = Responder(retriever, client=client, model="modello-custom")

    responder.answer("domanda")

    assert client.calls[0]["model"] == "modello-custom"