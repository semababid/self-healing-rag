# Experiments

Results from tuning retrieval (roadmap phase 9). All runs used `qwen3.5:9b`
through Ollama with thinking off and temperature 0, on an Apple M5 with 16 GB,
with `MAX_RETRIES=2` and `k=4` unless noted.

## Test set

29 questions in `src/evaluate.py`:

- 6 answerable questions worded close to the documents
- 18 answerable questions worded differently from the documents ("How long
  until a battery pack wears out?" for "rated for 500 charge cycles")
- 5 questions the documents don't answer. Four of them ask about the N7's
  colours, price, maximum height and battery type, which the documents give
  for the N5 and N9 but not for the N7, so a wrong answer here means the
  model mixed up the products.

A case passes if an answerable question gets an answer the critic accepts and
that contains the expected fact, or if an unanswerable question ends in the
"I don't have enough information" reply.

Each setting was run once. With temperature 0 the model mostly repeats itself,
but a difference of one or two questions between runs could still be noise.

## Corpus

The first six documents were too easy: with 16 chunks and k=4, a search
returned a quarter of the corpus, and only 1 of 18 reworded questions missed
the right chunk. Three documents were added on purpose to compete with the
real answers: a datasheet for an older model (N5), a preview of a newer model
(N9) and a docks and accessories guide. They give colours, prices, heights,
battery types and a second warranty period for products other than the N7.

## Retrieval only

`python -m src.evaluate --retrieval` checks where the chunk holding each
answer ranks, without calling the model. Each answerable case has an
`evidence` string, a phrase from the right chunk. "Top 4" counts answers
found in the 4 chunks the pipeline actually uses; MRR is the mean of 1/rank
(1.0 = always first).

Every chunk starts with its document's title (see "Chunk titles" below):

| chunk size / overlap | chunks | search | top 3 | top 4 | top 5 | MRR |
|---|---|---|---|---|---|---|
| 300 / 50 | 53 | vector | 20/24 | 22/24 | 22/24 | 0.78 |
| 300 / 50 | 53 | hybrid | 22/24 | 23/24 | 23/24 | 0.81 |
| 500 / 80 | 32 | vector | 23/24 | 23/24 | 23/24 | 0.84 |
| 500 / 80 | 32 | hybrid | 22/24 | 23/24 | 23/24 | 0.86 |
| 800 / 0 | 20 | vector | 21/24 | 22/24 | 22/24 | 0.69 |
| 800 / 0 | 20 | hybrid | 21/24 | 22/24 | 24/24 | 0.76 |
| 800 / 120 | 20 | vector | 20/24 | 21/24 | 22/24 | 0.69 |
| 800 / 120 | 20 | hybrid | 21/24 | 23/24 | 24/24 | 0.75 |
| 1200 / 150 | 13 | vector | 20/24 | 22/24 | 22/24 | 0.72 |
| 1200 / 150 | 13 | hybrid | 18/24 | 22/24 | 24/24 | 0.76 |

By this measure, 500-character chunks are best and hybrid search is at least
as good as vector search in most settings. The full runs below show that this
measure alone picks the wrong setting.

## Full runs

| run | chunks | search | titles | passed | wrong product | rescued by a retry | time |
|---|---|---|---|---|---|---|---|
| baseline | 800 / 120 | vector | no | 29/29 | 0 | 1 | 239 s |
| A | 500 / 80 | hybrid | no | 25/29 | 2 | 0 | 355 s |
| B | 800 / 120 | vector | yes | 29/29 | 0 | 3 | 286 s |
| C | 500 / 80 | vector | yes | 27/29 | 1 | 1 | 308 s |
| D | 500 / 80 | hybrid | yes | 28/29 | 0 | 0 | 267 s |

"Wrong product" counts unanswerable N7 questions answered with N5 or N9 facts.
"Rescued by a retry" counts answerable questions that were answered correctly
only after a query rewrite.

Runs E to H each change one thing from run B:

| run | change from B | passed | rescued by a retry | time |
|---|---|---|---|---|
| E | hybrid search | 29/29 | 4 | 371 s |
| F | chunk filter on | 27/29 (28/29 rescored) | 2 | 305 s |
| G | keep chunks on retry | 29/29 | 3 | 293 s |
| H | citations in answers | 29/29 | 2 | 253 s |

- Hybrid search (E) passed everything but needed four rescues instead of
  three, so its first answer was wrong more often, and it was 30% slower.
- The chunk filter (F) dropped the chunk with the holiday carry-over rule, so
  that question was refused, and the filter step itself took 29% of the run
  time. Its other failure was a fault in the test: the answer "covers
  manufacturing defects but not water damage or crashes caused by manual
  override" is right but didn't contain the expected words "not covered". The
  check now looks for "not" and "manual override" separately, which gives F
  28/29.
- Keeping chunks across retries (G) changed nothing measurable. Only a few
  questions get retried at all, so there is little for it to improve.
- Citations (H) cost nothing measurable: same score, and all 24 accepted
  answers cited their chunks, e.g. "Nimbus Care+ costs 349 EUR [4]".

## Chosen settings

800-character chunks with 120 overlap, vector search, k=4, chunk titles and
citations on, chunk filter and keeping chunks off. That is run H: 29/29 in
253 s.

Hybrid search, the chunk filter and keeping chunks are still available
(`SEARCH=hybrid`, `FILTER_CHUNKS=1`, `KEEP_CHUNKS=1`), since a smaller model
might benefit from them. The roadmap has that as 12.8.

## Findings

### Chunk titles

Run A gave the N9's price (21,500 EUR) and battery type (LFP) as the N7's, and
the critic accepted both. The chunk with those facts was the N9's "Planned
specifications" section, cut off from the document's title. It never said
"N9", but it did say "N7" ("Customers who trade in N7 drones get 15% off"),
which also made keyword search rank it highly for N7 questions. The generator
and the critic were misled by the same missing context.

`ingest.py` now starts every chunk with its document's title. In run D, the
same setting as A plus titles, both mix-ups were gone.

### Better ranking didn't mean better answers

500-character chunks ranked the right chunk higher than 800-character ones,
but answered worse: 27 and 28 of 29 against 29. With smaller chunks each one
carries less surrounding text, so the model gets less to work with even when
the right chunk is found. Run C still gave the N9's price once, with titles.
The retrieval benchmark is useful for quick comparisons, but settings have to
be confirmed with full runs.
