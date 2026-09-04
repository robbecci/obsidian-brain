"""Step 3: pipeline completa di indicizzazione.

Mette insieme gli step 1-3: parsing della vault -> chunking -> embedding
e persistenza in Chroma. Punto di ingresso comodo da usare da CLI o,
più avanti, dalla API FastAPI (step 6).
"""

from __future__ import annotations

from pathlib import Path

from obsidian_brain.chunker import chunk_notes
from obsidian_brain.vault_parser import load_vault
from obsidian_brain.vector_store import VectorStore


def build_index(
    vault_path: str | Path,
    persist_dir: str | Path = "chroma_db",
    reset: bool = False,
) -> VectorStore:
    """Esegue l'intera pipeline: legge la vault, la spezza in chunk,
    calcola gli embedding e li salva nel vector store.

    Args:
        vault_path: percorso della vault Obsidian da indicizzare.
        persist_dir: dove salvare il database Chroma su disco.
        reset: se True, cancella l'indice esistente prima di ricostruirlo
            (utile per evitare chunk orfani dopo aver rinominato/cancellato
            note nella vault).
    """
    store = VectorStore(persist_dir=persist_dir)
    if reset:
        store.reset()

    notes = load_vault(vault_path)
    chunks = chunk_notes(notes)
    store.add_chunks(chunks)

    print(f"Indicizzate {len(notes)} note in {len(chunks)} chunk ({store.count()} totali nell'indice).")
    return store