# Issue canonicalizzazione delle entità (2026-09-03) — RISOLTE

> **Stato: tutte e cinque corrette.** Interventi e verifica in fondo al file
> (§ *Correzioni applicate*). Le correzioni agiscono **prima** della scrittura:
> il grafo esistente va rigenerato perché abbiano effetto.


Diagnosi condotta sul campione ALCE `7139732942062154674`
(*"Which political party is currently in power in ireland?"*), 5 passaggi,
50 triple estratte e 50 scritte su Neo4j. Tutti i numeri riportati qui sotto
sono **misurati**, non stimati: sono l'output delle funzioni reali di
`src/ingestion/entity_canonicalizer.py` e degli embedding di
`settings.PREDICATE_EMBEDDING_MODEL`.

Fonti dei dati: `data/outputs/canonicalization.jsonl`,
`data/outputs/triples_extracted.jsonl` e il grafo Neo4j attuale.

---

## Sintesi

Il problema segnalato (`in Ireland` non viene unificato con
`Republic of Ireland`) è reale, ma **non è il più grave**. La diagnosi ha
fatto emergere cinque difetti distinti, di cui due producono *fusioni
sbagliate* già presenti nel grafo — un danno peggiore di una fusione mancata,
perché una fusione mancata perde recall mentre una fusione sbagliata
inquina la precision e non è più distinguibile a valle.

| # | Difetto | Direzione | Dove sta la causa |
|---|---------|-----------|-------------------|
| 1 | `in Ireland` ≠ `Republic of Ireland` | fusione **mancata** | stadio 1 (normalizzazione) |
| 2 | `Fine Gael` → `Fianna Fáil and Fine Gael` | fusione **sbagliata** | estrattore + stadio 2 |
| 3 | `seats` → `20 seats`, `the government` → `coalition government` | fusione **sbagliata** | stadio 2 |
| 4 | Nodi generici (`the country`, `fourth place`, …) | nodo **inattribuibile** | guardrail, a monte |
| 5 | `external_id` perde il disambiguatore Wikipedia | collisione **potenziale** | stadio 1 + stadio 3 |

---

## Issue 1 — `in Ireland` non si unifica con `Republic of Ireland`

### Sintomo

Nel grafo convivono due nodi distinti che denotano la stessa entità:

```
(Fianna Fáil) -[largest party from 1930s until 2011]-> (in Ireland)
(Labour Party) -[is opposition party in]-> (Republic of Ireland)
```

### Causa — tutti e quattro gli stadi mancano il bersaglio

```
stadio 1  normalize_mention('in Ireland') = 'in Ireland'
stadio 2  is_token_containment('in Ireland', 'Republic of Ireland') = False
          lexical_similarity                                        = 0.621
stadio 3  is_title_containment('in Ireland', 'Republic of Ireland') = False
stadio 4  cosine('in Ireland', 'Republic of Ireland')               = 0.645
```

Soglie attive: `CANONICALIZATION_LEXICAL_THRESHOLD = 0.90`,
`ENTITY_CLUSTER_THRESHOLD = 0.90`.

Nel dettaglio:

* **Stadio 1.** `_LEADING_ARTICLE` (`entity_canonicalizer.py:84`) rimuove solo
  `the|a|an`. La preposizione iniziale sopravvive, quindi la chiave di identità
  resta `in ireland` invece di `ireland`.
* **Stadio 2.** `is_token_containment` richiede che l'insieme dei token della
  forma corta sia **sottoinsieme** di quello della forma lunga. Il token `in`
  non compare in `{republic, of, ireland}`, quindi il test fallisce. La
  similarità lessicale di ripiego (0.621) è molto sotto la soglia.
* **Stadio 3.** Stesso motivo: `is_title_containment` prova prima il
  containment (fallisce) e poi il prefisso (`['republic','of'] != ['in','ireland']`).
* **Stadio 4.** 0.645 contro una soglia di 0.90.

### Perché NON si risolve abbassando le soglie

Il divario è **strutturale, non di calibrazione**. Portare la soglia a 0.645
farebbe passare anche coppie che devono restare separate:

```
cosine('the country', 'the state')          = 0.637
cosine('the country', 'Ireland')            = 0.624
cosine('Republic of Ireland', 'the country') = 0.533
```

`the country` e `the state` a 0.637 finirebbero fusi insieme a `in Ireland`
in un unico nodo-calderone.

### Correzione proposta

Estendere lo stadio 1 con lo strip delle preposizioni iniziali
(`in`, `on`, `at`, `of`, `to`, `from`, `by`, `for`, …). È deterministico,
idempotente e non tocca nessuna soglia. Dopo lo strip lo stadio 2 risolve da solo:

```
is_token_containment('Ireland', 'Republic of Ireland') = True   ✔ verificato
```

Effetto collaterale desiderabile: anche `in 2011` → `2011`.

Nota: lo stadio 3 resterebbe comunque in astensione, perché la menzione
`Ireland` combacia con **due** titoli dello scope
(`Republic of Ireland` e `Politics of the Republic of Ireland`) e
`TitleLinker` per progetto non linka quando c'è ambiguità. È corretto così:
la fusione la fa lo stadio 2, che è più a monte e più affidabile.

---

## Issue 2 — Fusione sbagliata: `Fine Gael` assorbito da una congiunzione

### Sintomo

Da `data/outputs/canonicalization.jsonl`:

```
'Fianna Fáil and Fine Gael'  ->  'Fianna Fáil and Fine Gael'   st=normalization
'Fine Gael'                  ->  'Fianna Fáil and Fine Gael'   st=lexical      ← FUSIONE SBAGLIATA
'Fianna Fáil'                ->  'Fianna Fáil'                 st=normalization
```

Il partito **Fine Gael** non esiste più come nodo autonomo: è diventato un
alias di un blob che denota due partiti insieme. **Fianna Fáil**, invece, è
rimasto nodo separato.

### Causa

`is_token_containment` (`entity_canonicalizer.py:121-148`) richiede che la
forma corta contenga la **testa** della forma lunga, cioè l'ultimo token.

```
is_token_containment('Fine Gael',   'Fianna Fáil and Fine Gael') = True
is_token_containment('Fianna Fáil', 'Fianna Fáil and Fine Gael') = False
```

In `fianna fáil and fine gael` l'ultimo token è `gael`, che appartiene a
`Fine Gael` ma non a `Fianna Fáil`. Da qui l'**asimmetria**: chi si trova per
caso in coda alla congiunzione viene inghiottito, chi sta in testa no.

Il vincolo sulla testa era stato introdotto proprio per evitare le valanghe
dell'union-find (`Apple` ⊂ `Apple iPhone` ⊂ `History of Apple Inc`), ma non
protegge dal caso della congiunzione, dove *entrambe* le parti sono entità
legittime e distinte.

### Gravità

Più alta della Issue 1. Una fusione mancata costa recall e resta
ispezionabile; una fusione sbagliata **produce un fatto falso nel grafo**
(`Fine Gael` risulta avere tutte le proprietà attribuite alla coppia) e
inquina in modo silenzioso la claim attribution.

### Correzione proposta

Due interventi, complementari:

1. **A monte, nei guardrail**: una menzione che contiene una congiunzione
   di coordinamento fra due entità nominate (`X and Y`, `X & Y`, `X, Y and Z`)
   non è un'entità. Va **scissa** in due triple con lo stesso predicato,
   oppure scartata con `discard_reason` dedicato.
2. **Nello stadio 2**: il containment non deve mai attraversare un confine
   di congiunzione. Se la forma lunga contiene `and`/`&`/`,` fra due token
   che sono entità nominate, il containment va rifiutato a prescindere.

---

## Issue 3 — La stessa regola sbaglia sui nomi comuni

Altre fusioni prodotte dallo stadio 2 sullo stesso campione:

```
'seats'                   ->  '20 seats'                                st=lexical
'the government'          ->  'coalition government'                    st=lexical
'recent general election' ->  'outcome of the recent general election'   st=lexical
```

Misurato:

```
is_token_containment('seats', '20 seats')                                          = True
is_token_containment('government', 'coalition government')                         = True
is_token_containment('recent general election',
                     'outcome of the recent general election')                     = True
```

L'ultima è chiaramente sbagliata sul piano semantico: **un'elezione non è
l'esito di un'elezione**. Le altre due fondono un nome comune generico con
una sua istanza specifica.

### Causa

`is_token_containment` filtra solo per lunghezza (`len(t) >= 3`) e per
`not t.isdigit()`. Non c'è nessun requisito di **nome proprio**: la regola
tratta `seats` e `government` come se fossero cognomi.

### Correzione proposta

Richiedere che la forma corta contenga almeno un token riconosciuto come
entità nominata. L'infrastruttura c'è già: `guardrails.EntityAnchorer`
(spaCy NER) è istanziato in `AlceIngestor._get_anchorer` e può essere
passato al canonicalizer. Senza almeno una NE, il containment non si applica
e la coppia scende agli stadi successivi.

---

## Issue 4 — Nodi generici: la causa vera della perdita di attribution

Nodi effettivamente scritti in Neo4j per il campione Irlanda:

```
the country -> country          the state -> state
third political party           coalition government
fourth place                    either of the two main parties
traditional centre ground       competing entities
opposition benches              independents
'comprising the islands of the country, Great Britain, the Isle of Man'
'enter negotiations with Fianna Fáil on forming a government'
'outcome of the recent general election'
```

Nessuno di questi è un'entità. Alcuni sono sintagmi nominali generici, altri
sono intere proposizioni finite nel campo oggetto.

### Causa

Il guardrail `generic_node` non li intercetta quando compaiono in posizione
di **oggetto**. Le triple superano il controllo e finiscono nel grafo.

### Gravità

È qui che muore la claim attribution, non nel tuning delle soglie 2/4. Un
claim generato dall'LLM non potrà mai agganciarsi a un nodo
`traditional centre ground`: l'exact match non trova nulla e il fallback
semantico confronta predicati fra nodi che non esistono nella risposta.
La canonicalizzazione **non può recuperare queste menzioni**, perché non
sono menzioni di entità: il posto giusto per fermarle è il guardrail.

Va anche notato che `the country` e `the state` denotano la Repubblica
d'Irlanda, esattamente come `in Ireland` e `Republic of Ireland`: senza il
guardrail restano tre nodi in più per la stessa entità, e a differenza di
`in Ireland` non sono recuperabili con una regola deterministica (sono
deittici risolvibili solo dal contesto, cioè un lavoro di coreferenza).

### Correzione proposta

Irrigidire `generic_node` sugli oggetti:

* rifiutare i sintagmi senza nessuna entità nominata e senza numero/data;
* rifiutare gli oggetti che superano una soglia di lunghezza in token
  (una proposizione non è un'entità);
* rifiutare i deittici generici residui (`the country`, `the state`,
  `the government`) — già coperti concettualmente da
  `unresolved_reference`, ma evidentemente non applicati in questa posizione.

---

## Issue 5 — `external_id` perde il disambiguatore Wikipedia

### Sintomo

Il passaggio `1907425` ha titolo `Labour Party (Ireland)`. L'`external_id`
prodotto è:

```
'Labour Party'  ->  external_id = 'wikipedia:Labour Party'
```

### Causa

`_PARENTHETICAL` (`entity_canonicalizer.py:75`) rimuove il contenuto fra
parentesi **prima** che il titolo venga usato come ID esterno
(`TitleLinker.link` → `f"wikipedia:{normalize_mention(title)}"`,
riga 245). Il disambiguatore `(Ireland)` — che è esattamente ciò che
distingue l'entità — viene buttato.

### Gravità

Innocua con `CANONICALIZATION_SCOPE=per_question` (default): dentro una
singola domanda non c'è un secondo Labour Party. Diventa un errore reale con
`scope=global`, dove il Labour Party irlandese e quello britannico
collasserebbero in un solo nodo, senza nessun segnale di allarme.

Poiché lo scope è un parametro dichiaratamente ablabile, il difetto va
corretto prima di eseguire l'ablazione, altrimenti il confronto
`per_question` vs `global` misurerebbe anche questo bug invece del solo
effetto dello scope.

### Correzione proposta

Separare le due cose, che oggi sono la stessa stringa:

* **`external_id`**: costruito sul titolo **integrale**, parentesi comprese
  (`wikipedia:Labour Party (Ireland)`) — è un identificatore, deve essere
  discriminante;
* **nome canonico visualizzato**: la forma normalizzata senza parentesi
  (`Labour Party`) — è un'etichetta, deve essere leggibile.

---

## Ablazione della soglia dello stadio 4 (misurata, 61 menzioni)

Prima di correggere è stata verificata l'ipotesi «basta abbassare la soglia».
Non basta, e costa carissimo:

```
0.90  47 nodi
0.80  47 nodi
0.75  46   + Fianna Fáil -> 'possible coalition of Fianna Fáil and the Progressive Democrats'
0.70  41   + Pat Rabbitte -> Labour Party            (una persona diventa un partito)
           + Progressive Democrats = Social Democrats (due partiti diversi)
           + 1 seat = 20 seats
           + second-largest party = third political party
0.65  37   + Supreme Court = The judiciary
0.60  35   + Republic of Ireland = in Ireland        (finalmente)
           + Chief Justice = Supreme Court = The judiciary
```

`in Ireland` arriva all'**ultimo gradino**, tre soglie dopo che Pat Rabbitte è
già diventato il Labour Party. E l'obiettivo vero non si raggiunge comunque:

```
cos('the country', 'Republic of Ireland') = 0.533
cos('the state',   'Republic of Ireland') = 0.385
```

Per fondere `the state` servirebbe ~0.38, cioè il collasso totale del grafo.

**Conclusione: nessuna soglia va toccata.** Risolvere `the country` →
`Republic of Ireland` è **coreferenza**, non similarità di superficie:
l'embedding di frase non può esprimerla. Il posto giusto è il prompt di
DeepSeek, che ha davanti il passaggio e il titolo.

## Ordine di intervento consigliato

1. **Issue 2** — guardrail sulle congiunzioni: ferma la corruzione attiva del
   grafo.
2. **Issue 4** — `generic_node` sugli oggetti: è il guadagno più grande sulla
   recall di attribution.
3. **Issue 1** — strip delle preposizioni iniziali nello stadio 1: tre righe,
   deterministico, nessuna soglia toccata.
4. **Issue 3** — vincolo di nome proprio nel containment dello stadio 2.
5. **Issue 5** — `external_id` con disambiguatore, prima dell'ablazione sullo
   scope.

**Nessuna soglia va toccata.** I numeri misurati mostrano che il divario di
0.645 contro 0.90 non è un problema di calibrazione: la banda 0.53–0.65 è la
stessa in cui vivono coppie che devono restare separate.

## Nota operativa

La canonicalizzazione avviene **prima** della scrittura
(`extract_entry` → `canonicalize_entry` → `write_entry`), quindi nessuna di
queste correzioni ha effetto retroattivo sul grafo esistente: dopo
l'implementazione serve una **re-ingestione completa** dei campioni già
processati (`force=True`, oppure `clear_graph` seguito da re-ingest).

Attenzione: `delete_by_source` cancella solo gli archi e lascia i nodi
`:Entity` orfani (nel grafo attuale ne è già presente uno,
`History of Apple Inc`). Un re-ingest con `force` va accompagnato dalla
cancellazione dei nodi a grado zero.

---

# Correzioni applicate

Principio guida, esplicitato anche nel prompt: **meno triple ma più precise**.
Lo stesso estrattore gira sui passaggi *e* sulla risposta generata, quindi un
nodo che non nomina nulla non potrà mai essere agganciato dall'attribution —
è peggio di nessuna tripla.

## A monte — prompt di estrazione (`deepseek_extractor.py`)

Il prompt vecchio autorizzava esplicitamente il problema:

```
2. Subject and object must be concrete named entities, dates, numbers
   or noun phrases as they appear in the passage.
   Never use pronouns (he, she, it, they, this) — always resolve them...
```

`or noun phrases` licenziava `traditional centre ground` e `fourth place`; il
divieto copriva **solo i pronomi**, e `the country` / `the state` sono
descrizioni definite, non pronomi. Regole nuove (esempi generici, mai casi del
corpus):

| Regola | Contenuto | Issue |
|--------|-----------|-------|
| 2 | S e O devono NOMINARE qualcosa; sintagma comune nudo mai accettato | 4 |
| 3 | Risolvere pronomi **e descrizioni definite**; se non si sa a chi si riferisce, scartare | 4 |
| 4 | Una entità per campo; coordinazione = più triple. Eccezione: nome proprio che contiene "and" | 2 |
| 5 | Oggetto senza preposizione iniziale: appartiene al predicato | 1 |
| 6 | Oggetto = entità, non proposizione | 4 |
| 10 | Lista vuota è una risposta valida e spesso corretta | — |

## Guardrail (`guardrails.py`) — la rete deterministica

Un prompt è probabilistico, un guardrail no. Due `REASONS` nuove:

* **`prepositional_object`** — l'oggetto inizia con una preposizione.
* **`conjunction_mention`** — S o O coordina due entità nominate distinte.
  Il discrimine è la **NER**, non la stringa: `EntityAnchorer.is_single_entity`
  (nuovo) riconosce `Procter and Gamble` e `Bosnia and Herzegovina` come nodo
  unico. `en_core_web_sm` però spacca `Trinidad and Tobago` in due GPE: è stato
  aggiunto il **titolo** come secondo segnale (match verbatim). Limite residuo
  accettato — la direzione di errore voluta è meno triple ma più precise.
  La coordinazione richiede una congiunzione **esplicita** (`and`/`&`): la
  virgola da sola spezzava `January 9, 2007` in due entità (regressione colta
  da un test esistente, non dal corpus).

Entrambe alimentano il **round di repair** già esistente: aggiungere una voce a
`REASON_HINTS` la trasforma in una correzione di secondo giro senza scrivere
logica nuova.

## Canonicalizzazione (`entity_canonicalizer.py`)

* **Issue 1** — `normalize_mention` rimuove in ciclo articoli **e preposizioni**
  iniziali (`"in the country"` → `"country"`). Lo strip vive anche qui, non solo
  nel guardrail, perché deve valere sui soggetti e su tutto lo storico già
  estratto. Costo accettato e documentato: i titoli che iniziano davvero con una
  preposizione perdono la testa (`"Of Mice and Men"` → `"Mice and Men"`) — è lo
  stesso compromesso già in essere per l'articolo, e la forma normalizzata è una
  **chiave di identità**, non il nome da mostrare (la menzione verbatim resta su
  `subject_surface` / `object_surface`).
* **Issue 2** — `is_token_containment` non attraversa mai una congiunzione.
* **Issue 3** — la forma corta deve contenere una maiuscola. Proxy
  deterministico di nome proprio: lo stadio 2 **deve restare deterministico**,
  quindi niente spaCy.
* **Issue 5** — `TitleLinker` costruisce l'ID sul titolo integrale
  (`wikipedia:Labour Party (Ireland)`), `_external_label` normalizza per
  l'etichetta del nodo (`Labour Party`). L'ID discrimina, l'etichetta si legge.

## UI (`app.py`)

`_render_neo4j_browser_commands`: dopo ogni scrittura la UI stampa le query
Cypher filtrate sui `source_id` del contesto appena inserito, più una sezione
di manutenzione (pulizia nodi orfani, controllo nodi/archi/passaggi).

Serve perché la vista di default del Browser — `MATCH (n) RETURN n LIMIT 25`,
senza `ORDER BY` — restituisce i nodi con **id interno più basso**, cioè i primi
mai scritti nel database. Un contesto appena inserito ha gli id più alti e in
quella vista non compare **mai**: il grafo sembra non essersi aggiornato mentre
invece lo è.

## Verifica

93 test passati (8 nuovi). Replay della cascata sulle **61 menzioni reali** del
campione Irlanda:

```
'in Ireland'            -> 'Republic of Ireland'      ✔ issue 1
'Fine Gael'             -> 'Fine Gael'                ✔ issue 2 (nodo autonomo)
'seats' / '20 seats'    -> nodi distinti              ✔ issue 3
'government' / 'coalition government' -> distinti     ✔ issue 3
'Labour Party'  ext='wikipedia:Labour Party (Ireland)' ✔ issue 5
```

Nodi 47 → **50**: quattro fusioni sbagliate annullate, una corretta aggiunta.
Il numero **sale** ed è il segno giusto: si stavano perdendo entità distinte.

Nota: nel replay `Fianna Fáil and Fine Gael` compare ancora come menzione,
perché il replay parte dalle triple **già estratte**. Sul nuovo run la
coordinazione non arriva più fin lì: la ferma `conjunction_mention` in fase di
estrazione.
