"""Step 3: persistenza degli embedding in Chroma (vector store locale).

VectorStore riceve un Embedder per dependency injection (non lo crea da
solo): questo rende possibile testare la logica di storage/retrieval con
un embedder finto, evitando di scaricare pesi di modello reali a ogni
esecuzione dei test.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from obsidian_brain.chunker import Chunk
from obsidian_brain.embedder import Embedder

DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION_NAME = "obsidian_vault"


class VectorStore:
    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedder: Embedder | None = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.embedder = embedder or Embedder()
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Calcola gli embedding per una lista di Chunk e li salva (o
        aggiorna, se un id esiste già - upsert) nel vector store.
        """
        if not chunks:
            return

        ids = [c.id for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = self.embedder.embed_texts(documents)
        metadatas = [self._chunk_to_metadata(c) for c in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    @staticmethod
    def _chunk_to_metadata(chunk: Chunk) -> dict:
        # Chroma non accetta liste come valori di metadata, solo tipi
        # primitivi (str, int, float, bool) - quindi header_path e tags
        # vengono appiattiti in stringhe.
        return {
            "note_title": chunk.note_title,
            "note_path": chunk.note_path,
            "header_path": " > ".join(chunk.header_path),
            "tags": ", ".join(chunk.tags),
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
        }

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        """Trova gli n_results chunk più simili semanticamente al testo
        dato. Restituisce una lista di dict con content, metadata e
        distance (più bassa = più simile), pronta per lo step 4
        (retrieval con soglia di rilevanza).
        """
        if self.count() == 0:
            return []

        query_embedding = self.embedder.embed_text(text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.count()),
        )
        return self._format_results(results)

    @staticmethod
    def _format_results(results: dict) -> list[dict]:
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            }
            for i in range(len(ids))
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Cancella tutti i dati della collection. Utile per re-indicizzare
        da zero dopo modifiche alla vault, senza duplicati residui.
        """
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(name=name)