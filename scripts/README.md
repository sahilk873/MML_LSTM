# Scripts Overview

This directory contains utility scripts around the main training pipeline.

## Core utilities

- `build_precomputed_embeddings.py`
  - Build precomputed embedding tables and the alias index used by ranking/scoring scripts.
- `rank_medic_pairs_rf.py`
  - Rank novel 2-drug MeDIC pairs for the disease panel using the saved RF model.
- `rank_vocab_pairs.py`
  - Rank CHEBI pairs directly from the saved model vocab using RF or pair-MLP scoring.
- `score_exact_triples.py`
  - Score an explicit list of drug-drug-disease triples with saved RF, pair-MLP, and LSTM models.
- `generate_novel_combos.py`
  - Generate and rank novel candidate combinations for a single target disease with the saved LSTM.

## Dataset and labeling utilities

- `rebuild_ground_truth_from_mechanisms.py`
  - Rebuild refined indications/contraindications datasets from mechanism labels.
- `classify_mechanisms.py`
  - Mechanism-labeling helper for source data curation.

## Experiment helpers

- `run_refined_training.py`
  - Convenience wrapper for a refined training run.
- `run_experiment_sweep.py`
  - Launch experiment sweeps over configurations.
- `compare_rf_old_new_holdout.py`
  - Compare RF outputs across old and new holdout workflows.
- `sweep_pairmlp_sigma.py`
  - Run sigma sweeps for pair-MLP initialization experiments.
- `sweep_pairmlp_sigma_replicates.py`
  - Repeated sigma sweep runs for stability checks.
- `train_triplet_lstm_baseline.py`
  - Train the triplet-based LSTM baseline from `train_kushal.parquet`.
- `eval_triplet_pair_metrics.py`
  - Evaluate triplet and aggregated pair metrics from triplet-model outputs.

## Inspection helpers

- `inspect_kg_relations.py`
  - Inspect relation structure in the KG source.
- `show_kg_triples.py`
  - Print or inspect KG triples for debugging.
- `test_one_row_classification.py`
  - Minimal one-row classification/debug helper.

For concrete commands, start with the root `README.md`.
