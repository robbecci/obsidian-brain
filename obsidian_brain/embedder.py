"""Step 3: generazione degli embedding.

Uso sentence-transformers invece di un'API esterna (es. Voyage AI) per una
ragione precisa in un progetto portfolio: chiunque clona il repository può
eseguirlo senza doversi procurare una chiave API. Il modello scelto,
paraphrase-multilingual-MiniLM-L12-v2, è multilingua e gestisce bene
l'italiano oltre che l'inglese - rilevante dato che le note della vault
sono in italiano.
"""


from __future__ import annotations

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    """Wrapper sottile attorno a sentence-transformers.

    Tenerlo separato da VectorStore permette di:
    - sostituire facilmente il modello (o passare a un'API come Voyage)
      senza toccare la logica di storage
    - testare VectorStore iniettando un Embedder finto, senza dover
      scaricare pesi di modello nei test
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Genera un embedding per ogni testo. Più efficiente di chiamare
        embed_text in un ciclo: sentence-transformers batcha internamente.
        """
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]