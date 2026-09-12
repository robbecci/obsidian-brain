"""Step 5: prompt grounding e generazione della risposta.

Uso un LLM locale via Ollama invece di un'API cloud: gira interamente
sul PC di chi usa il progetto, senza chiave API, senza costi, senza
bisogno di connessione internet una volta scaricato il modello. Il
compromesso e' che serve installare Ollama e scaricare i pesi del
modello (qualche GB), un'operazione una tantum - documentata nel README.

Principio cardine del progetto: il modello deve rispondere SOLO usando
il contesto recuperato dalla vault (step 4). Se il retrieval non trova
nulla di sufficientemente pertinente, non chiamiamo nemmeno il modello -
garantisce che non ci sia MAI un'allucinazione quando la vault non sa
rispondere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from obsidian_brain.retriever import Retriever

DEFAULT_MODEL = "llama3.2"

NOT_FOUND_MESSAGE = (
    "Non ho trovato informazioni sufficientemente pertinenti nella vault "
    "per rispondere a questa domanda."
)

SYSTEM_PROMPT = """Sei un assistente che risponde a domande basandosi ESCLUSIVAMENTE sul contesto fornito qui sotto, estratto da una vault Obsidian.

Regole rigide:
- Rispondi SOLO usando informazioni presenti nel contesto. Non usare conoscenza esterna, anche se la conosci.
- Se il contesto non contiene abbastanza informazioni per rispondere con sicurezza, dillo esplicitamente invece di indovinare o integrare con conoscenza generale.
- Quando rispondi, cita la nota di origine (indicata come "Fonte" nel contesto) da cui hai preso l'informazione.
- Sii conciso e diretto."""


class ChatClient(Protocol):
    """Interfaccia minima che un client di chat deve rispettare: riceve una
    lista di messaggi in stile OpenAI/Ollama ([{"role":..., "content":...}])
    e restituisce il testo della risposta. Astrarre dietro questa interfaccia
    permette di sostituire Ollama con qualsiasi altro backend (o un client
    finto nei test) senza toccare la logica di Responder.
    """

    def chat(self, model: str, messages: list[dict]) -> str: ...


class OllamaClient:
    """Wrapper sottile attorno al pacchetto ollama, che comunica con il
    demone Ollama in esecuzione in locale (di default su localhost:11434,
    avviato automaticamente dopo l'installazione).
    """

    def __init__(self):
        import ollama  # import qui dentro, cosi' il modulo resta
        # importabile anche senza il pacchetto ollama installato, quando
        # si inietta un client finto (es. nei test).
        self._ollama = ollama

    def chat(self, model: str, messages: list[dict]) -> str:
        response = self._ollama.chat(model=model, messages=messages)
        return response["message"]["content"]


@dataclass
class Answer:
    text: str
    found: bool
    sources: list[str]


class Responder:
    """Collega Retriever (step 4) al modello locale, applicando il
    principio di grounding: nessuna chiamata al modello, nessuna
    risposta, se il contesto non e' pertinente.
    """

    def __init__(
        self,
        retriever: Retriever,
        client: ChatClient | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.retriever = retriever
        self.client = client or OllamaClient()
        self.model = model

    def answer(self, query: str) -> Answer:
        result = self.retriever.retrieve(query)

        if not result.found:
            return Answer(text=NOT_FOUND_MESSAGE, found=False, sources=[])

        context = result.as_context()
        sources = sorted({c["metadata"]["note_title"] for c in result.chunks})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contesto:\n\n{context}\n\nDomanda: {query}"},
        ]
        text = self.client.chat(model=self.model, messages=messages)
        return Answer(text=text, found=True, sources=sources)