# MML Polypharmacy Runbook

This repository contains the polypharmacy modeling code for:

- LSTM-based drug-combination prediction
- RF and pair-MLP pair scoring on top of embedding features
- mixed-negative dataset construction
- exact triple scoring and repurposing/ranking workflows

The code lives in the repo. Large generated outputs should stay local and do not need to be committed.

## Repository layout

| Path | Purpose |
| --- | --- |
| `train.py` | Train the main LSTM classifier on drug-drug-disease examples. |
| `evaluate.py` | Re-evaluate saved model runs from an output directory. |
| `experiment.py` | Comparative experiment driver for LSTM, RF, and pairwise models. |
| `generalize.py` | Hold out a drug-count bucket and test generalization. |
| `polypharmacy/` | Core package: config, data loading, KG handling, models, utilities, triplet helpers. |
| `scripts/` | Utility scripts for preprocessing, ranking, scoring, sweeps, and auxiliary experiments. |
| `tests/` | Unit tests for the main pipeline and utility scripts. |
| `EXPERIMENTS.md` | Running record of experiments, metrics, and conclusions. |
| `AGENTS.md` | Repo instructions for agents, including the requirement to update `EXPERIMENTS.md`. |

## Environment setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the repo root:

```bash
PYTHONPATH=. .venv/bin/python <script>.py ...
```

## Required inputs

The core workflows assume these files are available locally:

- `artifacts/refined_gt/refined_indications.csv`
- `contraindications_norm_dedup.csv`
- `twosides_ddi_prefixed_normalized.csv`
- `kg_edges.parquet`

For precomputed-embedding workflows, also provide:

- `artifacts/refined/kg_embeddings.npz` or another saved embedding file
- `artifacts/precomputed_embeddings/topological/node_ids.npy`
- `artifacts/precomputed_embeddings/topological/embeddings.npy`
- `artifacts/precomputed_embeddings/topological/equivalent_id_to_node_id.parquet`

For MeDIC ranking workflows, also provide:

- `MeDIC Drug List.csv`
- `disease_codes_reference.md`

## Main workflows

### 1. Train the main LSTM model

```bash
PYTHONPATH=. .venv/bin/python train.py \
  --indications artifacts/refined_gt/refined_indications.csv \
  --contraindications contraindications_norm_dedup.csv \
  --kg kg_edges.parquet \
  --kg-embeddings artifacts/refined/kg_embeddings.npz \
  --twosides-contraindications twosides_ddi_prefixed_normalized.csv \
  --enable-mixed-negatives \
  --random-negative-ratio 1.0 \
  --random-negative-strategy disease_shuffle \
  --save-mixed-dataset-details \
  --output-dir artifacts/refined_train_precomputed
```

Common useful flags:

- `--config <json>` to override model/training defaults
- `--quick` for a smoke test
- `--disease-token-position first|last|none`
- `--concat-disease-after-lstm true|false`

### 2. Run the comparative experiment

This is the main driver for the saved mixed-negative experiments in this repo.

```bash
PYTHONPATH=. .venv/bin/python experiment.py \
  --indications artifacts/refined_gt/refined_indications.csv \
  --contraindications contraindications_norm_dedup.csv \
  --kg kg_edges.parquet \
  --kg-embeddings artifacts/refined/kg_embeddings.npz \
  --twosides-contraindications twosides_ddi_prefixed_normalized.csv \
  --enable-mixed-negatives \
  --random-negative-ratio 1.0 \
  --random-negative-strategy disease_shuffle \
  --save-mixed-dataset-details \
  --output-dir artifacts/exp_refined_mixed_twosides_topological512
```

Typical outputs:

- `best_model.pt`
- `rf_model.pkl`
- `pair_mlp_best.pt`
- `metrics.json`
- `mixed_negative_report.json`

### 3. Re-evaluate a saved run

```bash
PYTHONPATH=. .venv/bin/python evaluate.py \
  --output-dir artifacts/exp_refined_mixed_twosides_topological512 \
  --indications artifacts/refined_gt/refined_indications.csv \
  --contraindications contraindications_norm_dedup.csv \
  --kg kg_edges.parquet \
  --twosides-contraindications twosides_ddi_prefixed_normalized.csv \
  --enable-mixed-negatives
```

### 4. Hold out a drug-count bucket

```bash
PYTHONPATH=. .venv/bin/python generalize.py \
  --indications artifacts/refined_gt/refined_indications.csv \
  --contraindications contraindications_norm_dedup.csv \
  --kg kg_edges.parquet \
  --kg-embeddings artifacts/refined/kg_embeddings.npz \
  --twosides-contraindications twosides_ddi_prefixed_normalized.csv \
  --enable-mixed-negatives \
  --holdout-drug-count 3-4 \
  --output-dir artifacts_refined_combo_holdout
```

Accepted `--holdout-drug-count` forms:

- `1`
- `2`
- `3-4`
- `>=5`

## Ranking and scoring workflows

### Build or refresh the alias index

```bash
PYTHONPATH=. .venv/bin/python scripts/build_precomputed_embeddings.py \
  --only-equivalent-id-index
```

### Rank MeDIC candidate pairs with the RF model

```bash
PYTHONPATH=. .venv/bin/python scripts/rank_medic_pairs_rf.py \
  --model-output-dir artifacts/exp_refined_mixed_twosides_topological512 \
  --rf-model-path artifacts/exp_refined_mixed_twosides_topological512/rf_model.pkl \
  --precomputed-node-ids artifacts/precomputed_embeddings/topological/node_ids.npy \
  --precomputed-embeddings artifacts/precomputed_embeddings/topological/embeddings.npy \
  --medic-drug-list "MeDIC Drug List.csv" \
  --alias-index artifacts/precomputed_embeddings/topological/equivalent_id_to_node_id.parquet \
  --disease-reference-md disease_codes_reference.md \
  --novelty-source deduped \
  --top-n 50 \
  --batch-size 200000 \
  --max-workers 4 \
  --output-dir artifacts/rf_repurpose_top50
```

### Rank CHEBI vocabulary pairs with RF or pair-MLP

```bash
PYTHONPATH=. .venv/bin/python scripts/rank_vocab_pairs.py \
  --model-type pair_mlp \
  --model-output-dir artifacts/exp_refined_mixed_twosides_topological512_pairmlp_low_sigma \
  --disease-reference-md disease_codes_reference.md \
  --novelty-source deduped \
  --top-n 50 \
  --output-dir artifacts/vocab_repurpose
```

### Score exact triples with saved models

`--triples-json` must point to a JSON list with entries like:

```json
[
  {
    "label": "example",
    "drug_ids": ["CHEBI:15365", "CHEBI:6801"],
    "disease_id": "MONDO:0005148"
  }
]
```

Run:

```bash
PYTHONPATH=. .venv/bin/python scripts/score_exact_triples.py \
  --triples-json path/to/triples.json \
  --model-output-dir artifacts/exp_refined_mixed_twosides_topological512 \
  --precomputed-node-ids artifacts/precomputed_embeddings/topological/node_ids.npy \
  --precomputed-embeddings artifacts/precomputed_embeddings/topological/embeddings.npy \
  --alias-index artifacts/precomputed_embeddings/topological/equivalent_id_to_node_id.parquet \
  --output-json scored_triples.json
```

### Generate novel candidate combinations with the saved LSTM

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_novel_combos.py \
  --model-output-dir artifacts/refined_train_precomputed \
  --target-disease MONDO:0005148 \
  --min-combo-size 2 \
  --max-combo-size 2 \
  --top-n 200 \
  --min-prob 0.9 \
  --output-dir artifacts_repurpose
```

## Other scripts

See `scripts/README.md` for a short summary of the utility scripts that are in the repo and when to use them.

## Tests

Run the core regression suite:

```bash
PYTHONPATH=. .venv/bin/python -m unittest \
  tests.test_mixed_negative_pipeline \
  tests.test_build_precomputed_embeddings \
  tests.test_rank_medic_pairs_rf \
  tests.test_generate_novel_combos \
  tests.test_rank_vocab_pairs \
  tests.test_triplet_metrics
```

## Notes

- RF and pairwise models use concatenated `[drug1_emb, drug2_emb, disease_emb]` features.
- The MeDIC ranking workflow resolves `curie` and `alternate_ids` through the alias index.
- `MONDO:0011699` is substituted to `MONDO:0005265` in the ranking workflow where IBD uses the latter embedding.
