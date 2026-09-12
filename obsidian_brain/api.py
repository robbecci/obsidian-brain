"""Step 6: layer API con FastAPI.

Espone Responder (step 5) come servizio HTTP. La costruzione dell'app e'
separata dalla costruzione dell'indice: create_app() riceve un Responder
gia' pronto (facile da testare con uno finto), mentre build_app() e' la
funzione "reale" che indicizza una vault e assembla tutto per l'uso
effettivo (vedi il blocco __main__ in fondo al file).
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from obsidian_brain.indexer import build_index
from obsidian_brain.responder import DEFAULT_MODEL as DEFAULT_OLLAMA_MODEL
from obsidian_brain.responder import Responder
from obsidian_brain.retriever import Retriever


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="La domanda da porre alla vault.")


class AskResponse(BaseModel):
    answer: str
    found: bool
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int


def create_app(responder: Responder) -> FastAPI:
    """Crea l'app FastAPI a partire da un Responder gia' pronto. Separare
    questa funzione da build_app() permette di testare gli endpoint
    iniettando un Responder finto, senza dover indicizzare una vault vera
    o avere Ollama in esecuzione a ogni run dei test.
    """
    app = FastAPI(
        title="Obsidian Brain API",
        description="Fai domande alla tua vault Obsidian. Risponde solo in base al contenuto indicizzato.",
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            chunks_indexed=responder.retriever.store.count(),
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        result = responder.answer(request.question)
        return AskResponse(answer=result.text, found=result.found, sources=result.sources)

    return app


def build_app(
    vault_path: str = "sample_vault",
    persist_dir: str = "chroma_db",
    model: str = DEFAULT_OLLAMA_MODEL,
    reset: bool = False,
) -> FastAPI:
    """Punto di ingresso 'reale': indicizza la vault, costruisce Responder,
    e restituisce l'app pronta per essere servita da uvicorn.
    """
    store = build_index(vault_path, persist_dir=persist_dir, reset=reset)
    responder = Responder(Retriever(store), model=model)
    return create_app(responder)


if __name__ == "__main__":
    # Avvio diretto: python -m obsidian_brain.api
    # Le variabili d'ambiente permettono di personalizzare vault/persist_dir
    # senza modificare il codice - utile per chi clona il progetto.
    import uvicorn

    app = build_app(
        vault_path=os.environ.get("VAULT_PATH", "sample_vault"),
        persist_dir=os.environ.get("PERSIST_DIR", "chroma_db"),
        model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)