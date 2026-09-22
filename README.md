# MML-LSTM: Polypharmacy Drug Combination & Repurposing Prediction

A research pipeline for predicting whether combinations of drugs are effective (indicated) or dangerous (contraindicated) treatments for a given disease. The project combines knowledge-graph embeddings with sequence models (LSTM) and pairwise scorers (Random Forest, MLP) to rank drug-combination candidates for repurposing.

## Motivation

Polypharmacy — treating a condition with multiple drugs simultaneously — is common in complex diseases but hard to reason about: drug interactions can be synergistic, neutral, or harmful depending on the disease context. This project frames the problem as **triple classification and ranking**: given a set of drugs and a disease, predict whether the combination is a plausible treatment, and rank candidate combinations for novel drug repurposing.

## Approach

1. **Knowledge graph embeddings** — drug and disease nodes are embedded via `node2vec` random walks over a biomedical knowledge graph, giving each entity a dense vector that encodes its relational context.
2. **Sequence model (LSTM)** — variable-length drug combinations are encoded with an LSTM over their embeddings; the disease embedding is concatenated with the final hidden state and passed through an MLP classifier to predict indication vs. contraindication.
3. **Pairwise baselines** — a Random Forest and a Pair-MLP are trained on concatenated `[drug1, drug2, disease]` embeddings as strong, simpler baselines for ranking candidate pairs.
4. **Triplet ranking model** — an alternative formulation that optimizes ranking (via triplet loss) rather than binary classification, evaluated with enrichment-factor metrics (EF5/EF10/EF20) suited to large candidate spaces.
5. **Mixed-negative training** — training sets combine hard contraindication negatives with randomly shuffled disease-negative pairs to improve robustness and generalization.

## Key results

| Model | Accuracy | AUROC | F1 | Notes |
| --- | ---: | ---: | ---: | --- |
| Random Forest (refined ground truth, shared holdout) | 0.888 | 0.985 | 0.907 | Best saved held-out classification baseline |
| Random Forest (prior ground truth, shared holdout) | 0.940 | 0.981 | 0.953 | Higher sensitivity, lower specificity than refined GT |
| Pair-MLP (sigma sweep, best AUROC) | — | 0.966 | — | Best diversity/accuracy tradeoff for repurposing candidates |
| Triplet LSTM (validation) | — | 0.992 | 0.893 | Strong discrimination on held-out triples |
| Triplet LSTM (large candidate space) | — | 0.78–0.81 | low (as classifier) | EF5=16.2, EF10=8.8, EF20=4.7 — 5–16x enrichment over random |

**Takeaway:** thresholded classification metrics degrade in very large candidate spaces (as expected — positives become rare), but ranking/enrichment metrics stay strong, which is why the pipeline treats these models primarily as **rankers** for drug-repurposing candidate generation rather than binary classifiers.

## Repository layout

| Path | Purpose |
| --- | --- |
| `train.py` | Train the main LSTM classifier on drug-drug-disease examples. |
| `evaluate.py` | Re-evaluate saved model runs from an output directory. |
| `experiment.py` | Comparative experiment driver for LSTM, RF, and pairwise models. |
| `generalize.py` | Hold out a drug-count bucket and test generalization. |
| `polypharmacy/` | Core package: config, data loading, KG handling, models, triplet helpers, utilities. |
| `scripts/` | Utility scripts for preprocessing, ranking, scoring, sweeps, and repurposing workflows (see `scripts/README.md`). |
| `tests/` | Unit tests covering the core pipeline and utility scripts. |

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

This repository contains code only — the biomedical datasets and knowledge graph used in this project are not redistributed here. To run the pipeline end-to-end you will need your own copies of:

- An indications/contraindications dataset (drug set + disease + label)
- A biomedical knowledge graph edge list (e.g. as a parquet file of edges)
- Optional: a drug-drug interaction dataset (e.g. TWOSIDES) for mixed-negative training

## Main workflows

### 1. Train the main LSTM model

```bash
PYTHONPATH=. .venv/bin/python train.py \
  --indications <indications.csv> \
  --contraindications <contraindications.csv> \
  --kg <kg_edges.parquet> \
  --enable-mixed-negatives \
  --random-negative-ratio 1.0 \
  --random-negative-strategy disease_shuffle \
  --output-dir artifacts/refined_train
```

Useful flags: `--config <json>` to override model/training defaults, `--quick` for a smoke test, `--disease-token-position first|last|none`, `--concat-disease-after-lstm true|false`.

### 2. Run the comparative experiment (LSTM vs. RF vs. Pair-MLP)

```bash
PYTHONPATH=. .venv/bin/python experiment.py \
  --indications <indications.csv> \
  --contraindications <contraindications.csv> \
  --kg <kg_edges.parquet> \
  --enable-mixed-negatives \
  --output-dir artifacts/experiment_run
```

Produces `best_model.pt`, `rf_model.pkl`, `pair_mlp_best.pt`, `metrics.json`, and `mixed_negative_report.json`.

### 3. Re-evaluate a saved run

```bash
PYTHONPATH=. .venv/bin/python evaluate.py \
  --output-dir artifacts/experiment_run \
  --indications <indications.csv> \
  --contraindications <contraindications.csv> \
  --kg <kg_edges.parquet> \
  --enable-mixed-negatives
```

### 4. Test generalization by holding out a drug-count bucket

```bash
PYTHONPATH=. .venv/bin/python generalize.py \
  --indications <indications.csv> \
  --contraindications <contraindications.csv> \
  --kg <kg_edges.parquet> \
  --enable-mixed-negatives \
  --holdout-drug-count 3-4 \
  --output-dir artifacts/combo_holdout
```

Accepted `--holdout-drug-count` forms: `1`, `2`, `3-4`, `>=5`.

## Ranking and repurposing scripts

See `scripts/README.md` for the full list. Highlights:

- `rank_medic_pairs_rf.py` / `rank_vocab_pairs.py` — rank novel drug-pair candidates for a disease using saved RF or Pair-MLP models.
- `score_exact_triples.py` — score an explicit list of drug-drug-disease triples with saved models.
- `generate_novel_combos.py` — generate and rank novel candidate combinations for a target disease with the saved LSTM.
- `build_precomputed_embeddings.py` — precompute embedding tables and alias indices used by the ranking scripts.

## Tests

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover tests
```

## Notes

- RF and pairwise models use concatenated `[drug1_emb, drug2_emb, disease_emb]` features.
- Ranking workflows resolve entity identifiers through an alias index built from the knowledge graph.
- Data files, trained model checkpoints, and generated artifacts are intentionally excluded from version control (see `.gitignore`); all workflows write outputs to a local `artifacts/` directory.
