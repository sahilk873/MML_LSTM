# MML Polypharmacy Runbook

This repository contains the polypharmacy training, evaluation, and repurposing workflow built around drug-drug-disease tuples, KG-derived embeddings, mixed negative construction, and downstream RF/LSTM ranking.

## Repo layout

| Area | Description |
| --- | --- |
| `train.py` / `evaluate.py` / `generalize.py` / `experiment.py` | Main training, evaluation, generalization, and comparative experiment entrypoints. |
| `polypharmacy/` | Shared data loading, model, KG, and utility code. |
| `scripts/` | Helper scripts for ground-truth rebuilding, embedding export, and RF-based ranking over MeDIC. |
| `artifacts/` | Model checkpoints, metrics, mixed-negative reports, and ranking outputs. |

## Environment

Create a virtualenv and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run scripts from repo root with `PYTHONPATH=.`:

```bash
PYTHONPATH=. .venv/bin/python <script>.py ...
```

## Core datasets

Main tuple datasets used in the current workflow:

- `artifacts/refined_gt/refined_indications.csv`
  - Mechanistically synergistic positive indications.
- `contraindications_norm_dedup.csv`
  - Deduplicated sourced contraindication negatives.
- `twosides_ddi_prefixed_normalized.csv`
  - Additional sourced DDI-style negative tuples.
- `MeDIC Drug List.csv`
  - Drug universe used for RF repurposing candidate generation.
- `disease_codes_reference.md`
  - Ten target diseases used in RF repurposing runs.

Embedding assets:

- `artifacts/refined/kg_embeddings.npz`
  - Precomputed KG embeddings used by training/eval runs.
- `artifacts/precomputed_embeddings/topological/node_ids.npy`
- `artifacts/precomputed_embeddings/topological/embeddings.npy`
- `artifacts/precomputed_embeddings/topological/equivalent_id_to_node_id.parquet`
  - Global precomputed node IDs, vectors, and alias index used for MeDIC-to-embedding matching.

## Mixed-negative training setup

The current training path supports:

- positives from `refined_indications.csv`
- sourced negatives from `contraindications_norm_dedup.csv`
- optional sourced negatives from `twosides_ddi_prefixed_normalized.csv`
- randomized disease-shuffled negatives at a configurable ratio

Shared mixed-negative flags are available in:

- `train.py`
- `generalize.py`
- `evaluate.py` fallback rebuild path
- `experiment.py`

Relevant flags:

- `--twosides-contraindications`
- `--enable-mixed-negatives`
- `--random-negative-ratio`
- `--random-negative-strategy`
- `--save-mixed-dataset-details`

## Training command used for the mixed-negative experiment

This is the main experiment configuration used for the refined positives + sourced negatives + TWOSIDES + randomized negatives setup:

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

Expected outputs in the run directory:

- `best_model.pt`
- `rf_model.pkl`
- `rf_model_metadata.json`
- `metrics.json`
- `mixed_negative_report.json`

## Precomputed embedding alias index

`scripts/build_precomputed_embeddings.py` now exports `equivalent_id_to_node_id.parquet` alongside the embedding arrays.

The alias index contains:

- `alias_id`
- `node_id`
- `match_source`

It is built from canonical `id` plus `equivalent_identifiers` from the source embedding parquet files and improves matching coverage for MeDIC `curie` and `alternate_ids`.

Backfill alias index only:

```bash
PYTHONPATH=. .venv/bin/python scripts/build_precomputed_embeddings.py \
  --only-equivalent-id-index
```

## RF repurposing ranking over MeDIC

`scripts/rank_medic_pairs_rf.py` does the following:

- maps `MeDIC Drug List.csv` drugs via `curie + alternate_ids`
- resolves aliases against `equivalent_id_to_node_id.parquet`
- uses the full topological precomputed embedding space
- scores all possible 2-drug combinations for the ten diseases in `disease_codes_reference.md`
- excludes known disease-specific combos from the training dataset
- writes top 50 RF-ranked novel pairs per disease plus a combined file
- adds `drug_name_1` and `drug_name_2` to every ranking CSV automatically
- supports disease-level parallelism with `--max-workers`

### Command to run everything

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

Outputs:

- `artifacts/rf_repurpose_top50/<run_name>/top50_all_diseases.csv`
- `artifacts/rf_repurpose_top50/<run_name>/top50_MONDO_*.csv`
- `artifacts/rf_repurpose_top50/<run_name>/medic_mapping_matched.csv`
- `artifacts/rf_repurpose_top50/<run_name>/medic_mapping_unmatched.csv`
- `artifacts/rf_repurpose_top50/<run_name>/disease_run_summary.csv`
- `artifacts/rf_repurpose_top50/<run_name>/summary.json`

## Tests added for recent changes

- `tests/test_mixed_negative_pipeline.py`
- `tests/test_build_precomputed_embeddings.py`
- `tests/test_rank_medic_pairs_rf.py`

Run targeted tests:

```bash
PYTHONPATH=. .venv/bin/python -m unittest \
  tests.test_mixed_negative_pipeline \
  tests.test_build_precomputed_embeddings \
  tests.test_rank_medic_pairs_rf
```

## Notes

- RF ranking uses the trained RF model with concatenated `[drug1_emb, drug2_emb, disease_emb]` features.
- The ranking workflow intentionally uses `curie + alternate_ids` only for MeDIC matching. `ingredient_ids` are excluded to avoid component-vs-product semantic drift.
- `MONDO:0011699` is substituted to `MONDO:0005265` for IBD because that is the disease embedding used in this workflow.
