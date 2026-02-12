# MML Polypharmacy Training Runbook

This repo trains a binary classifier for:
- `1` = indication (therapeutic/safe)
- `0` = contraindication (unsafe/adverse)

The main scripts in active use are:
- `train.py`
- `evaluate.py`
- `scripts/classify_mechanisms.py`
- `scripts/rebuild_ground_truth_from_mechanisms.py`
- `scripts/run_refined_training.py`

## 1) Environment setup

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

Use `PYTHONPATH=.` when running scripts from repo root:

```bash
PYTHONPATH=. .venv312/bin/python <script>.py ...
```

## 2) Baseline experiment (original dedup labels)

Train:

```bash
PYTHONPATH=. .venv312/bin/python train.py \
  --indications indications_norm_dedup.csv \
  --contraindications contraindications_norm_dedup.csv \
  --kg kg_edges.parquet \
  --output-dir artifacts_baseline
```

Evaluate:

```bash
PYTHONPATH=. .venv312/bin/python evaluate.py \
  --indications indications_norm_dedup.csv \
  --contraindications contraindications_norm_dedup.csv \
  --kg kg_edges.parquet \
  --output-dir artifacts_baseline
```

## 3) Mechanism labeling with OpenAI (new workflow)

Set API key:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

### A. Indications-only mechanism labeling (recommended path used here)

If you want to keep all contraindications as fixed negatives and only relabel/filter positives:

```bash
# create empty contraindications CSV with same headers
PYTHONPATH=. .venv312/bin/python - <<'PY'
import pandas as pd
cols = pd.read_csv('contraindications_norm_dedup.csv', nrows=0).columns
pd.DataFrame(columns=cols).to_csv('artifacts/mechanism_labels/contraindications_empty.csv', index=False)
PY
```

Run mechanism classification:

```bash
OPENAI_API_KEY="$OPENAI_API_KEY" PYTHONPATH=. .venv312/bin/python scripts/classify_mechanisms.py \
  --indications-csv indications_norm_dedup.csv \
  --contraindications-csv artifacts/mechanism_labels/contraindications_empty.csv \
  --output-dir artifacts/mechanism_labels \
  --model gpt-4.1-mini \
  --prompt-version v1
```

### B. (Optional) Label both indications + contraindications

```bash
OPENAI_API_KEY="$OPENAI_API_KEY" PYTHONPATH=. .venv312/bin/python scripts/classify_mechanisms.py \
  --indications-csv indications_norm_dedup.csv \
  --contraindications-csv contraindications_norm_dedup.csv \
  --output-dir artifacts/mechanism_labels \
  --model gpt-4.1-mini \
  --prompt-version v1
```

## 4) Build refined ground truth from mechanism labels

Keep only mechanistically synergistic positives with confidence >= 0.6, and drop rows marked for review:

```bash
PYTHONPATH=. .venv312/bin/python scripts/rebuild_ground_truth_from_mechanisms.py \
  --labeled-dataset-csv artifacts/mechanism_labels/mechanism_labeled_dataset.csv \
  --output-dir artifacts/refined_gt \
  --keep-categories mechanistically_synergistic \
  --min-confidence 0.6 \
  --drop-needs-review
```

## 5) Train refined experiment while keeping original contraindications

```bash
PYTHONPATH=. .venv312/bin/python train.py \
  --indications artifacts/refined_gt/refined_indications.csv \
  --contraindications contraindications_norm_dedup.csv \
  --kg kg_edges.parquet \
  --output-dir artifacts_refined
```

Evaluate refined:

```bash
PYTHONPATH=. .venv312/bin/python evaluate.py \
  --indications artifacts/refined_gt/refined_indications.csv \
  --contraindications contraindications_norm_dedup.csv \
  --kg kg_edges.parquet \
  --output-dir artifacts_refined
```

## 6) Where mechanism categories are stored

- Summary counts: `artifacts/mechanism_labels/classification_summary.json`
- Row-level labels: `artifacts/mechanism_labels/mechanism_annotations.csv`
- Merged dataset: `artifacts/mechanism_labels/mechanism_labeled_dataset.csv`

Categories:
- `mechanistically_synergistic`
- `side_effect_relief`
- `common_comorbidity`
- `other`

