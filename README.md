# MML Polypharmacy Runbook

This repository hosts the full polypharmacy classification workflow from raw deduped U.S. indications/contraindications tables through KG-enhanced LSTM training, plus OpenAI-assisted mechanism relabeling and manual review artifacts.

## Repo layout at a glance

| Area | Description |
| --- | --- |
| `train.py` / `evaluate.py` | Core training/evaluation scripts. They rely on `polypharmacy/data.py`, `polypharmacy/kg.py`, and `polypharmacy/model.py`. |
| `scripts/` | Helpers for LLM classification (`classify_mechanisms.py`), rebuilding refined targets, running comparative experiments, etc. |
| `polypharmacy/` | Shared modules (data loaders, KG utils, LSTM model, OpenAI classifier prompt). |
| `artifacts/` | Output directories. Keep only the runners you care about; others live under `artifacts_archive/`. |

## Essentials before running

1. Install dependencies:

   ```bash
   python3.12 -m venv .venv312
   source .venv312/bin/activate
   pip install -r requirements.txt
   ```

2. Run every script from repo root with `PYTHONPATH=.` so local modules resolve correctly:

   ```bash
   PYTHONPATH=. .venv312/bin/python <script>.py ...
   ```

3. Primary inputs:
   - `indications_norm_dedup.csv` and `contraindications_norm_dedup.csv` (normalized drug sets + MONDO diseases).
   - `kg_edges.parquet` (knowledge graph edges used for embeddings).

## Step-by-step workflow

### 1) Baseline train/eval (original dedup dataset)

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

### 2) Mechanism relabeling with OpenAI

1. Export your OpenAI key:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. Prepare an empty contraindications CSV if you want to treat current negatives as fixed:

   ```bash
   PYTHONPATH=. .venv312/bin/python - <<'PY'
   import pandas as pd
   cols = pd.read_csv('contraindications_norm_dedup.csv', nrows=0).columns
   pd.DataFrame(columns=cols).to_csv('artifacts/mechanism_labels/contraindications_empty.csv', index=False)
   PY
   ```

3. Run the OpenAI classifier (uses `polypharmacy/llm_classifier.py` which now targets `gpt-5-mini` without unsupported params):

   ```bash
   OPENAI_API_KEY="$OPENAI_API_KEY" PYTHONPATH=. .venv312/bin/python scripts/classify_mechanisms.py \
     --indications-csv indications_norm_dedup.csv \
     --contraindications-csv artifacts/mechanism_labels/contraindications_empty.csv \
     --output-dir artifacts/mechanism_labels \
     --model gpt-5-mini \
     --prompt-version v1 \
     --workers 8 \
     --force
   ```

4. Optional: to relabel both indications and contraindications, point `--contraindications-csv` at `contraindications_norm_dedup.csv` instead of the empty CSV.

### 3) Build refined ground truth

```bash
PYTHONPATH=. .venv312/bin/python scripts/rebuild_ground_truth_from_mechanisms.py \
  --labeled-dataset-csv artifacts/mechanism_labels/mechanism_labeled_dataset.csv \
  --output-dir artifacts/refined_gt \
  --keep-categories mechanistically_synergistic \
  --min-confidence 0.6 \
  --drop-needs-review
```

Outputs:

- `artifacts/refined_gt/refined_indications.csv`: new positives (train schema).
- `artifacts/refined_gt/refined_contraindications.csv`: labeled negatives (often empty for indication-only runs).
- `artifacts/refined_gt/refined_dataset.csv` + `refinement_report.json`: dedup + filtering stats.

### 4) Train with refined positives and original negatives

Use precomputed KG embeddings (`artifacts_refined/kg_embeddings.npz`) to skip expensive node2vec:

```bash
PYTHONPATH=. .venv312/bin/python train.py \
  --indications artifacts/refined_gt/refined_indications.csv \
  --contraindications contraindications_norm_dedup.csv \
  --kg kg_edges.parquet \
  --kg-embeddings artifacts_refined/kg_embeddings.npz \
  --output-dir artifacts_refined
```

Evaluate the best checkpoint from that run:

```bash
PYTHONPATH=. .venv312/bin/python evaluate.py \
  --indications artifacts/refined_gt/refined_indications.csv \
  --contraindications contraindications_norm_dedup.csv \
  --kg kg_edges.parquet \
  --output-dir artifacts_refined
```

### 5) Useful helper scripts

- `scripts/run_refined_training.py`: trains both baseline (
`artifacts_baseline`) and refined runs (`artifacts_refined`) and writes a comparison report.
- `scripts/test_one_row_classification.py`: smoke-test a single row via OpenAI (requires `OPENAI_API_KEY`).

## Key outputs and navigation

| File | Description |
| --- | --- |
| `artifacts/redo_20260221_161114/mechanism_labels/mechanism_labeled_dataset.csv` | Full LLM-labeled dataset (drug sets, categories, rationale). |
| `classification_summary.json` | Aggregated counts + failure stats. |
| `mechanism_annotations.csv` | Row-level log of classifications. |
| `category_examples_30_each_manual_review_format.xlsx` | Manual-review workbook (set + classification). |
| `artifacts/refined_gt/refined_indications.csv` | Training positives after filtering + dedup. |
| `artifacts_refined/best_model.pt` | Refined-trained checkpoint (uses cached KG embeddings). |

## Artifacts housekeeping

- Keep only one “gold” workspace (e.g., `artifacts/redo_20260221_161114/`).
- Archive outdated runs under `artifacts_archive/<date>/...` (see `2026-02-redo-cutover`).
- Raw KG embeddings and manual reviews sit under `artifacts/precomputed_embeddings/` and `artifacts/redo_20260221_161114/mechanism_labels/` respectively.

## Troubleshooting & tips

| Problem | Notes |
| --- | --- |
| `ModuleNotFoundError: polypharmacy` | Always run with `PYTHONPATH=.` or install the package via `pip install -e .`. |
| OpenAI `temperature` error | Fixed already—`polypharmacy/llm_classifier.py` no longer sets `temperature`. |
| Node2vec permission issues | Use `--kg-embeddings artifacts_refined/kg_embeddings.npz` to skip running node2vec. |
| Need deterministic splits? | Splits are saved to `<output-dir>/splits.npz` inside each `artifacts_*` run. |

## Next steps

1. Update `polypharmacy/llm_classifier.py`’s `SYSTEM_PROMPT` if you want to try a new prompt version (e.g., `v2`).
2. Use `scripts/run_refined_training.py` to benchmark new variants against the baseline.
3. Keep `artifacts_archive/2026-02-redo-cutover/README.md` updated every time you retire an old run.
