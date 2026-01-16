# Project State

## Summary
- End-to-end polypharmacy prediction pipeline implemented (data load, KG node2vec embeddings, LSTM+MLP model, deterministic splits, training, evaluation).
- Artifacts saved under `artifacts/` (deduped/filtered datasets, dropped rows, KG embeddings, vocabularies, splits, checkpoints).

## Key Data Assumptions
- `primary_drug_id_norm` and `secondary_drug_id_norm` are stringified Python lists and are parsed via `ast.literal_eval`.
- Drug sets are constructed as `sorted(primary_drug_id_norm + secondary_drug_id_norm)`.
- Drug list entries are normalized by flattening one level and coercing to strings before sorting.
- Labels are inferred from source file: indications = 1, contraindications = 0.
- Only normalized IDs are used for modeling (CHEBI for drugs, MONDO for diseases).

## Model Architecture Choices
- Drug embeddings initialized from KG node2vec vectors (optional freeze).
- Variable-length drug sequences encoded with a single-layer LSTM; final hidden state is the drug-combo embedding.
- Disease embedding concatenated after LSTM output and fed to an MLP classifier.
- Binary classification with a single logit output (BCEWithLogitsLoss).

## Constraints and Invariants
- Deduplicate on `(drug_set, condition_id_norm)`; if conflicts exist, assign label = 0.
- KG coverage filtering drops any example with drugs/disease missing from KG nodes.
- Deterministic splits are required; splits are reused only if dataset size matches.
- Saved artifacts are considered authoritative for evaluation.
- The training run dataset is saved to `artifacts/filtered_dataset_run.csv` and drives splits/evaluation.

## Change Log
- 2026-01-12: Added deduplication + conflict resolution after CSV load, persisted deduped dataset and class counts, and integrated KG coverage filtering with dropped-row reporting to enforce KG node availability and determinism.
- 2026-01-12: Normalized drug ID lists by flattening nested list entries and coercing to strings to prevent sorting errors during drug set construction.
- 2026-01-12: Added quick-run CLI controls (sampling edges/examples + override epochs/batch/walks) and persisted the run dataset for deterministic evaluation.
- 2026-01-12: Guarded against empty datasets/splits during quick runs; added clearer errors and NaN-safe evaluation when validation/test splits are empty.
- 2026-01-12: Added explicit errors when train/val/test splits are empty after sampling so the user can adjust sampling controls instead of hitting a PyTorch error.
- 2026-01-12: Pre-filter KG edges to only nodes present in the dataset before running node2vec and keep using `--kg-workers` so walk generation uses multiple CPU cores without rebuilding alias tables.
- 2026-01-12: Added optional KG hop expansion so pruning can include k-hop neighbors (with node caps and verbose hop logging) before node2vec runs.
- 2026-01-12: Extended evaluation metrics to include confusion matrix, specificity, sensitivity, and F1 so we can better characterize performance beyond AUC/accuracy.
- 2026-01-12: Added `experiment.py` to compare a Random Forest (drug+drug+disease embeddings) vs. the LSTM using a fixed 2-drug test set, ensuring both models see the same held-out triples.
