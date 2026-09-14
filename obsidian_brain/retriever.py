"""Step 4: retrieval con soglia di rilevanza.

Principio cardine del progetto: il sistema deve rispondere SOLO se ha
trovato contenuto sufficientemente pertinente nella vault. Se la distanza
tra la domanda e il miglior chunk trovato supera una soglia, la vault
"non lo sa" - meglio ammetterlo che azzardare una risposta su un match
debole (che allo step 5 porterebbe a possibili allucinazioni).
"""

from __future__ import annotations

from dataclasses import dataclass

from obsidian_brain.vector_store import VectorStore

# Validato empiricamente allo step 7 su una vault con note ricche di
# contesto (paragrafi multi-frase, non singole righe): su un campione di
# 11 domande, le domande pertinenti avevano distanza top-1 <= 0.582,
# quelle fuori tema >= 0.763 - un margine di sicurezza di quasi 0.18 in
# cui 0.6 si colloca comodamente. Con note molto brevi (poche parole) il
# margine si riduce o sparisce: la qualità del retrieval dipende quindi
# anche da quanto contesto contengono le note originali, non solo dai
# parametri di chunking.
DEFAULT_MAX_DISTANCE = 0.6
DEFAULT_N_RESULTS = 5


@dataclass
class RetrievalResult:
    found: bool
    chunks: list[dict]  # sottoinsieme dei risultati che superano la soglia

    def as_context(self) -> str:
        """Concatena i chunk trovati in un blocco di testo con citazione
        della fonte, pronto per essere iniettato nel prompt allo step 5.
        """
        parts = []
        for c in self.chunks:
            meta = c["metadata"]
            source = meta["note_title"]
            if meta.get("header_path"):
                source += f" > {meta['header_path']}"
            parts.append(f"[Fonte: {source}]\n{c['content']}")
        return "\n\n---\n\n".join(parts)


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        n_results: int = DEFAULT_N_RESULTS,
    ):
        self.store = store
        self.max_distance = max_distance
        self.n_results = n_results

    def retrieve(self, query: str) -> RetrievalResult:
        raw_results = self.store.query(query, n_results=self.n_results)
        relevant = [r for r in raw_results if r["distance"] <= self.max_distance]

        if not relevant:
            return RetrievalResult(found=False, chunks=[])
        return RetrievalResult(found=True, chunks=relevant)