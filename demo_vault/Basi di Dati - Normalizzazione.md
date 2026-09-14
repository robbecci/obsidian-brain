---
title: Basi di Dati - Normalizzazione
tags: [università, database, sql]
---

## Perché normalizzare

La normalizzazione è il processo con cui si organizzano i dati di un database relazionale per ridurre al minimo la ridondanza e prevenire alcune anomalie che possono verificarsi durante l'inserimento, l'aggiornamento o la cancellazione dei dati. Senza normalizzazione, la stessa informazione può finire ripetuta in più righe della stessa tabella, con il rischio che un aggiornamento venga applicato a una copia ma non alle altre, lasciando il database in uno stato incoerente.

## Le forme normali principali

La prima forma normale (1NF) richiede che ogni colonna contenga un solo valore atomico, cioè non divisibile ulteriormente, e che non esistano gruppi di colonne ripetute. La seconda forma normale (2NF) richiede che la tabella sia già in 1NF e che ogni attributo non chiave dipenda dall'intera chiave primaria, non solo da una sua parte. La terza forma normale (3NF) va oltre: richiede che non esistano dipendenze transitive, cioè che un attributo non chiave non dipenda da un altro attributo non chiave.

## Un esempio pratico

Immaginiamo una tabella Ordini che contiene, per ogni riga, il nome del cliente e il suo indirizzo insieme ai dettagli dell'ordine. Se lo stesso cliente effettua dieci ordini, il suo indirizzo viene ripetuto dieci volte. Normalizzando, si crea una tabella separata Clienti, collegata alla tabella Ordini tramite una chiave esterna: l'indirizzo viene scritto una sola volta, e se il cliente si trasferisce basta aggiornare una singola riga.

Vedi anche [[Chiavi Esterne]] per approfondire il concetto di relazione tra tabelle.