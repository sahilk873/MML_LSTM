# Polypharmacy Kedro

Kedro pipeline that reproduces the polypharmacy indication vs contraindication workflow using
normalized CHEBI (drugs) and MONDO (diseases) identifiers plus KG-derived node2vec embeddings.

## Data layout

Place the inputs in `data/01_raw/`:

- `indications_norm.csv`
- `contraindications_norm.csv`
- `kg_edges.parquet`

Optional single-therapy CSV paths can be configured in `conf/base/parameters.yml`.

## Install dependencies

```
pip install -r requirements.txt
```

## Run the pipeline

```
kedro run
```

The default pipeline is `polypharmacy` and produces:

- `data/03_primary/deduped_dataset.csv`
- `data/03_primary/filtered_dataset.csv`
- `data/04_feature/filtered_dataset_run.csv`
- `data/05_model_input/*` (encoded sequences + splits)
- `data/06_models/*` (embeddings, vocab, checkpoint, config)
- `data/08_reporting/*` (metrics and stats)

## Evaluate with the standalone script

After `kedro run`, you can evaluate the saved checkpoint:

```
python scripts/evaluate.py
```

## Configuration

All model, training, and KG settings live in `conf/base/parameters.yml` under the
`polypharmacy` key. This includes options for precomputed KG embeddings and optional
single-therapy CSV paths.
