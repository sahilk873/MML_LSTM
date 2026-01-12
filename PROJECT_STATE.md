# Project State

## Summary
- End-to-end polypharmacy prediction pipeline implemented (data load, KG node2vec embeddings, LSTM+MLP model, deterministic splits, training, evaluation).
- Artifacts saved under `artifacts/` (deduped/filtered datasets, dropped rows, KG embeddings, vocabularies, splits, checkpoints).

## Key Data Assumptions
- `primary_drug_id_norm` and `secondary_drug_id_norm` are stringified Python lists and are parsed via `ast.literal_eval`.
- Drug sets are constructed as `sorted(primary_drug_id_norm + secondary_drug_id_norm)`.
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

## Change Log
- 2026-01-12: Added deduplication + conflict resolution after CSV load, persisted deduped dataset and class counts, and integrated KG coverage filtering with dropped-row reporting to enforce KG node availability and determinism.
