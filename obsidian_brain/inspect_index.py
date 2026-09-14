"""Step 7: strumento diagnostico per calibrare il sistema su una vault reale.

A differenza di sample_vault (4 note brevi, scritte apposta per testare
casi limite), una vault reale ha note di lunghezza e stile molto più
variabili. Le costanti scelte finora (DEFAULT_CHUNK_SIZE, DEFAULT_MAX_DISTANCE)
erano stime ragionevoli ma non calibrate su dati veri - questo script aiuta
a farlo in modo sistematico:

1. Mostra statistiche sul chunking (quante note, quanti chunk, dimensioni)
   per capire se DEFAULT_CHUNK_SIZE produce chunk troppo piccoli/grandi.
2. Per un elenco di domande vere, mostra le distanze coseno GREZZE (senza
   applicare la soglia) verso i chunk più vicini - così si vede a occhio
   se 0.6 è troppo permissivo, troppo severo, o giusto.

Uso:
    python -m obsidian_brain.inspect_index --vault "C:\\percorso\\della\\tua\\vault"
    python -m obsidian_brain.inspect_index --vault MyVault --questions domande.txt
"""

from __future__ import annotations

import argparse
import statistics

from obsidian_brain.chunker import chunk_notes
from obsidian_brain.indexer import build_index
from obsidian_brain.vault_parser import load_vault


def print_chunk_stats(vault_path: str) -> None:
    notes = load_vault(vault_path)
    chunks = chunk_notes(notes)

    if not chunks:
        print("Nessun chunk generato. La vault è vuota o illeggibile?")
        return

    sizes = [c.char_count for c in chunks]
    multi_chunk_notes = sum(1 for n in notes if sum(1 for c in chunks if c.note_path == n.relative_path) > 1)

    print("=== Statistiche chunking ===")
    print(f"Note trovate:              {len(notes)}")
    print(f"Chunk generati:            {len(chunks)}")
    print(f"Note divise in più chunk:  {multi_chunk_notes}")
    print(f"Dimensione chunk (char):   min={min(sizes)}  media={statistics.mean(sizes):.0f}  "
          f"mediana={statistics.median(sizes):.0f}  max={max(sizes)}")
    print()


def print_query_distances(vault_path: str, questions: list[str], persist_dir: str, n_results: int = 5) -> None:
    print("=== Distanze coseno grezze (senza soglia) ===")
    store = build_index(vault_path, persist_dir=persist_dir, reset=True)
    print()

    for q in questions:
        print(f"Domanda: {q}")
        results = store.query(q, n_results=n_results)
        for r in results:
            print(f"  {r['distance']:.3f}  -  {r['metadata']['note_title']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Diagnostica chunking e retrieval su una vault reale.")
    parser.add_argument("--vault", required=True, help="Percorso della vault da analizzare.")
    parser.add_argument("--persist-dir", default="chroma_db_diagnostics", help="Cartella temporanea per l'indice.")
    parser.add_argument(
        "--questions",
        help="File di testo con una domanda vera per riga. Se omesso, mostra solo le statistiche di chunking.",
    )
    parser.add_argument("--n-results", type=int, default=5, help="Quanti risultati mostrare per domanda.")
    args = parser.parse_args()

    print_chunk_stats(args.vault)

    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
        print_query_distances(args.vault, questions, args.persist_dir, args.n_results)
    else:
        print("Nessun file di domande fornito (--questions). Solo statistiche di chunking mostrate.")


if __name__ == "__main__":
    main()