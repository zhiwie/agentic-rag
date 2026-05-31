# Evaluation

Measures retrieval **accuracy** and **performance** with real numbers.

## Run it
```bash
python -m eval.run_eval
```

With `OPENAI_API_KEY` set, it uses production embeddings (record these numbers
for your slide). Without a key, it runs on the offline FakeEmbedder to prove the
harness works.

## What it reports
- **Hit@1 / Hit@3** — fraction of questions whose correct source document was the
  top result / among the top 3.
- **MRR** (Mean Reciprocal Rank) — averages 1/rank of the first correct hit;
  rewards ranking the right document highly.
- **Latency** — average and p95 retrieval time per query.

## Files
- `eval_set.json` — 10 labeled questions, each tagged with the document that
  should be retrieved and keywords a correct answer should contain.
- `run_eval.py` — the harness.

## Latest offline run (FakeEmbedder baseline)
Hit@1 = 70% · Hit@3 = 90% · MRR = 0.825 · ~0.8 ms/query.
Production OpenAI embeddings are expected to improve accuracy further — re-run
with your key to populate final figures.
