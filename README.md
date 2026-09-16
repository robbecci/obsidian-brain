# 🧠 Obsidian Brain

Sistema RAG (Retrieval-Augmented Generation) **locale e gratuito** che risponde a domande basandosi esclusivamente sul contenuto di una vault Obsidian — mai su conoscenza esterna, mai inventando. Se la vault non contiene informazioni sufficienti, il sistema lo dice esplicitamente invece di allucinare.

Progetto realizzato come lavoro di portfolio nell'ambito di un percorso di AI Development.

## Come funziona

L'utente pone una domanda (via API HTTP o da codice Python). Il sistema cerca nella vault indicizzata i frammenti di testo più semanticamente vicini alla domanda; solo se ne trova di sufficientemente pertinenti, li passa a un modello linguistico locale che formula una risposta basata **esclusivamente** su quel contenuto, citando la nota di origine. Se nessun frammento è abbastanza pertinente, il sistema rifiuta di rispondere invece di inventare.

```
Vault Obsidian (.md)
        │
        ▼
[1] Parsing (frontmatter, wikilink, tag)
        │
        ▼
[2] Chunking (per sezione Markdown, con overlap)
        │
        ▼
[3] Embedding locale (sentence-transformers) + ChromaDB
        │
        ▼
[4] Retrieval con soglia di rilevanza (distanza coseno)
        │
        ▼
[5] Generazione risposta (Ollama, locale) — solo se il contesto è pertinente
        │
        ▼
[6] Esposto via API (FastAPI)
```

## Stack tecnologico

- **Python 3.14** — linguaggio principale
- **sentence-transformers** — embedding testuale multilingua, eseguito in locale
- **ChromaDB** — vector store locale, persistente su disco
- **Ollama** (`llama3.2`) — generazione della risposta, modello linguistico locale
- **FastAPI** — layer API HTTP, con documentazione interattiva automatica
- **pytest** — 47 test automatici, tutti con dipendenze finte (nessun test richiede rete, Ollama, o modelli scaricati)

## Risultati della calibrazione

La soglia di rilevanza del retrieval (`DEFAULT_MAX_DISTANCE = 0.6`) è stata validata empiricamente su un campione di 11 domande reali: le domande pertinenti avevano una distanza coseno migliore ≤ 0.582, quelle fuori tema ≥ 0.763 — un margine di sicurezza di quasi 0.18 in cui la soglia si colloca comodamente, con zero falsi positivi e zero falsi negativi sul campione testato.

## Come avviarlo

Istruzioni complete nel documento `Obsidian_Brain_Guida_Configurazione.pdf` incluso nella repository. In breve:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
ollama pull llama3.2
python -m pytest tests/ -v
python -m obsidian_brain.api
```

Poi apri il browser su `http://127.0.0.1:8000/docs` per la documentazione interattiva dell'API.

## Documentazione

- 📄 `Obsidian_Brain_Documentazione_Tecnica.pdf` — scelte tecniche, architettura, pipeline dettagliata, bug reali risolti durante lo sviluppo, calibrazione, limiti noti e sviluppi futuri
- 📄 `Obsidian_Brain_Guida_Configurazione.pdf` — istruzioni dettagliate di installazione ed esecuzione, con risoluzione dei problemi più comuni

## Struttura del progetto

```
obsidian_brain/
├── vault_parser.py     # step 1: parsing frontmatter, wikilink, tag
├── chunker.py           # step 2: divisione in chunk con header path
├── embedder.py            # step 3: embedding locale
├── vector_store.py         # step 3: persistenza ChromaDB, distanza coseno
├── retriever.py              # step 4: retrieval con soglia di rilevanza
├── responder.py                # step 5: generazione risposta (Ollama)
├── api.py                        # step 6: layer HTTP (FastAPI)
├── indexer.py                      # pipeline completa
└── inspect_index.py                  # strumento diagnostico di calibrazione

tests/            # 47 test automatici
sample_vault/     # fixture minimale usata dai test
demo_vault/       # vault di esempio più ricca, per dimostrazioni realistiche
```

## Limiti noti

- La qualità del retrieval dipende dalla ricchezza di contesto delle note: con note fatte di frasi singole molto brevi, il margine tra domande pertinenti e non pertinenti si riduce (vedi documentazione tecnica, sezione 7)
- L'indice viene ricostruito da zero a ogni riavvio: accettabile per vault di dimensioni moderate, da rivedere per vault molto grandi
- Un solo modello di embedding multilingua generico: per un dominio molto specialistico un modello fine-tuned potrebbe fare meglio
