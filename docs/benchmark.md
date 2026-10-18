# Benchmark

## Task

Given a job title, return the NOC 2021 unit group it belongs to. That's what an applicant has in hand.

## Data

Statistics Canada lists 27,935 example job titles, each under exactly one of the 516 unit groups ("All
examples" in the NOC 2021 V1.0 elements file).

## Split

A title goes to the test set when the SHA-256 of its text is divisible by 5:

- **5,468 test titles** covering 506 unit groups; 22,467 stay in the index
- The split depends only on the text: no random seed, same result on every machine
- Test titles are removed from the index, so no method can find a test title by looking it up
- Every unit group keeps at least one example title in the index

Near-duplicates stay on both sides on purpose (`web programmer` held out, `web developer` indexed), because
real users type titles that resemble the official ones without matching them.

## Metrics

| Metric | Meaning |
|---|---|
| top-1 | the first answer is right |
| top-5 | the right code is among the first five, the list a user actually reads |
| MRR | mean reciprocal rank of the right code within the top 10 |
| confusable error share | of the wrong first answers, the share that Statistics Canada's own exclusions list as easy to confuse with the right one |

The last one separates "the method is bad" from "the classification itself is ambiguous here".

## Results

Reproduce with `uv run noc-noc eval`.

| Method | top-1 | top-5 | MRR | Confusable errors |
|---|---|---|---|---|
| TF-IDF, word 1–2-grams | 65.6% | 88.3% | 0.753 | 37% |
| Embeddings, one vector per unit group | 56.1% | 81.2% | 0.671 | 34% |
| Embeddings, one vector per title and duty | 68.9% | 93.5% | 0.795 | 38% |
| **Hybrid: TF-IDF + per-passage embeddings (RRF)** | **72.2%** | **93.6%** | **0.816** | 41% |

Embeddings are `BAAI/bge-small-en-v1.5` through fastembed (ONNX, CPU). Embedding a unit group's whole text
once is *worse* than TF-IDF: the model truncates long texts, and one vector blurs 50 job titles together.
Embedding each title and duty separately and scoring a unit group by its best passage works like a nearest
neighbour over the indexed titles, and beats TF-IDF. Fusing both rankings with reciprocal rank fusion
(k = 60) is best. As accuracy rises, a larger share of the remaining errors are pairs Statistics Canada
itself lists as easy to confuse.

### What goes into a unit group's text (TF-IDF)

| Fields | top-1 | top-5 |
|---|---|---|
| title + example titles | 60.9% | 85.9% |
| + definition | 63.3% | 86.9% |
| + main duties | **65.6%** | **88.3%** |
| + title repeated three times | 65.5% | 88.3% |

Character 3–5-grams scored lower than words (56.8% top-1), despite tolerating typos.

### Reranking with Claude

`claude-opus-5-5` at low effort reranks the hybrid's top 10 and cites the official duties that match, by id
(`21232-d3`). Codes outside the candidates, repeated codes and duty ids that don't belong to the code are
dropped; candidates the model leaves out keep their hybrid order.

**Not measured yet**: it needs an API key, and scoring all 5,468 titles would cost about $90. Compare it with
the hybrid on the same fixed sample of 200 titles (about $3.50):

```bash
uv run noc-noc eval --method hybrid --method rerank --sample 200
```

## Second benchmark: real titles from Job Bank

The official example titles are clean. Employers write things like `*Project Engineer (2 Positions) - North
Shore Wastewater Treatment Plant`. Canada's Job Bank publishes every posting with the title as written and the
NOC 2021 code it was filed under
([dataset](https://open.canada.ca/data/en/dataset/ea639e28-c0fc-48bf-b5dd-b8899bd43072), Open Government
Licence – Canada).

From the July–September 2026 files (about 146,000 postings), a title is kept when:

- the posting's official language is English
- the title isn't `NA` (about two thirds of postings) and the code is a NOC 2021 unit group
- the title isn't an official example title or unit group title, so no method can look it up
- every posting with that title was filed under the same code

That leaves **4,242 distinct titles over 315 unit groups**, committed in
[`src/noc_noc/data/job_bank_2026_07_09.tsv`](../src/noc_noc/data/job_bank_2026_07_09.tsv). The index is the full
classification. Caveats: the codes were chosen by employers, so some labels are wrong, and about 3% of the
titles are in French despite the language filter.

| Method | top-1 | top-5 | MRR | Confusable errors |
|---|---|---|---|---|
| TF-IDF | 51.1% | 72.6% | 0.604 | 25% |
| Embeddings, per title and duty | 58.7% | 79.3% | 0.677 | 29% |
| **Hybrid** | **59.1%** | **79.4%** | **0.679** | 27% |

Real titles are about 13 points harder than official ones. The methods keep their order, and embeddings gain
more over TF-IDF on messy titles (+7.6 points of top-1, against +3.3 on official titles).

Reproduce with `uv run noc-noc eval --benchmark job-bank`.
