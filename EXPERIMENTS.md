# Experiment Record

This file is the durable experiment log for this repository.

Rules for updating it:

- Add a new entry whenever a new experiment run produces results worth keeping.
- Prefer recorded numbers from saved `metrics.json`, `summary.json`, `metrics_summary.json`, or equivalent output files.
- If an experiment is exploratory or incomplete, say that explicitly instead of pretending it is final.
- When a newer experiment supersedes an older one, keep both and say which one became the new baseline.

## Canonical metric conventions

- Classification metrics are read from the saved run artifacts.
- Pair-model metrics usually refer to the fixed held-out 2-drug test split in `splits.npz`.
- Ranking diversity metrics refer to uniqueness across the exported top-50 recommendations per disease.

## Data curation and ground-truth changes

### Refined mechanism-ground-truth curation

Source: `artifacts/refined_gt/refinement_report.json`

- Input rows: `5762`
- Rows after category filter: `4337`
- Rows after confidence filter: `4324`
- Rows after review filter: `4057`
- Final refined rows after dedup: `3601`
- Keep category: `mechanistically_synergistic`
- Drops:
  - category: `1425`
  - confidence: `13`
  - needs_review: `267`

Interpretation:

- The refined-positive dataset is materially smaller than the older indication set.
- The refined set is higher precision by construction, but it reduces positive coverage substantially.

## Main model experiments

### Baseline mixed-negative experiment

Artifact: `artifacts/exp_refined_mixed_twosides`

Setup:

- Refined positives
- Contraindication negatives
- TWOSIDES negatives
- Random disease-shuffle negatives

Held-out metrics:

| Model | Accuracy | AUROC | F1 | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.9030 | 0.9607 | 0.8681 | 0.8229 | 0.9538 |
| RF | 0.9556 | 0.9921 | 0.9421 | 0.9323 | 0.9703 |

Takeaway:

- RF clearly outperformed the LSTM on this run and became the early strong pair-ranking baseline.

### Topological-512 embedding experiment

Artifact: `artifacts/exp_refined_mixed_twosides_topological512`

Held-out metrics:

| Model | Accuracy | AUROC | F1 | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.9440 | 0.9647 | 0.8776 | 0.8658 | 0.9676 |
| RF | 0.9502 | 0.9895 | 0.8824 | 0.8054 | 0.9939 |

Takeaway:

- Moving to the topological-512 embedding setup improved the LSTM materially.
- RF still had the best overall AUROC and specificity.

### Topological-512 with alias-aware mapping

Artifact: `artifacts/exp_refined_mixed_twosides_topological512_alias`

Held-out metrics:

| Model | Accuracy | AUROC | F1 | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.9498 | 0.9748 | 0.8791 | 0.8418 | 0.9797 |
| Pair MLP | 0.9461 | 0.9705 | 0.8736 | 0.8588 | 0.9703 |
| RF | 0.9559 | 0.9923 | 0.8896 | 0.8192 | 0.9937 |

Takeaway:

- This is the strongest saved topological pair-model comparison in the repo.
- RF remained the best pure held-out classifier.
- LSTM and Pair-MLP were competitive but not better than RF on AUROC.

### Pair-MLP low-sigma / slow-learning-rate experiment

Artifact: `artifacts/exp_refined_mixed_twosides_pairmlp_low_sigma_slow_lr`

Held-out metrics:

| Model | Accuracy | AUROC | F1 | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.9091 | 0.9707 | 0.8831 | 0.8854 | 0.9241 |
| Pair MLP | 0.9152 | 0.9647 | 0.8889 | 0.8750 | 0.9406 |
| RF | 0.9556 | 0.9921 | 0.9421 | 0.9323 | 0.9703 |

Takeaway:

- The low-sigma Pair-MLP was viable, but not enough to beat RF on held-out classification.
- This run later became useful mainly for diversity-oriented ranking analysis.

### Requested-triples experiments

Artifacts:

- `artifacts/exp_requested_triples_topological512_direct_lstm`
- `artifacts/exp_requested_triples_topological512_exact`

Saved metrics match the `artifacts/exp_refined_mixed_twosides_topological512_pairmlp_low_sigma` family:

| Model | Accuracy | AUROC | F1 |
| --- | ---: | ---: | ---: |
| LSTM | 0.9362 | 0.9684 | 0.8571 |
| Pair MLP | 0.9425 | 0.9651 | 0.8711 |
| RF | 0.9502 | 0.9895 | 0.8824 |

Interpretation:

- These artifacts appear to reuse the same fixed test-set performance envelope while focusing on exact-triple scoring workflows rather than a new classification regime.

## Old-vs-new ground-truth comparison

Artifact: `artifacts/rf_old_vs_new_holdout/summary.json`

Shared holdout:

- Common keys: `2237`
- Holdout keys: `447`

RF results on shared holdout:

| Dataset | Accuracy | AUROC | F1 | Precision | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old GT | 0.9396 | 0.9813 | 0.9532 | 0.9615 | 0.9450 | 0.9295 |
| New refined GT | 0.8881 | 0.9852 | 0.9071 | 0.9879 | 0.8385 | 0.9808 |

Delta `new - old`:

- Accuracy: `-0.0515`
- AUROC: `+0.0039`
- F1: `-0.0461`
- Sensitivity: `-0.1065`
- Specificity: `+0.0513`

Takeaway:

- The refined GT made the RF model more conservative.
- Precision and specificity improved, but recall/sensitivity dropped.
- This is consistent with the refined set favoring stricter positives.

## Repurposing and ranking experiments

### Alias-aware MeDIC repurposing comparison

Artifacts:

- `artifacts/repurpose_top50_alias/rf/summary.json`
- `artifacts/repurpose_top50_alias/pair_mlp/summary.json`
- `artifacts/repurpose_top50_alias/lstm/summary.json`
- `artifacts/repurpose_top50_alias/lstm_no_menthol/summary.json`

Shared setup:

- Mapped MeDIC drugs: `2513`
- Mapped MeDIC drugs present in precomputed embeddings: `2513` for RF/Pair-MLP/LSTM, `2512` for `lstm_no_menthol`
- Unmatched MeDIC drugs: `513`
- Top results exported: `500` total rows across 10 diseases

Uniqueness summary:

| Model | Unique pairs | Reused pair rows | Pairs reused across multiple diseases |
| --- | ---: | ---: | ---: |
| RF | 150 | 431 | 81 |
| Pair MLP | 427 | 135 | 62 |
| LSTM | 160 | 435 | 95 |
| LSTM no menthol | 168 | 434 | 102 |

Takeaway:

- RF and LSTM produced heavily reused recommendations across diseases.
- Pair-MLP produced far more diverse candidate pairs.
- Excluding menthol did not fundamentally fix the LSTM reuse pattern.

### Pair-MLP uniqueness analysis

Artifact: `artifacts/pairmlp_uniqueness_analysis/summary.json`

Direct comparison:

| Model | Exported rows | Unique pairs | Reused pair rows | Max disease reuse for a single pair |
| --- | ---: | ---: | ---: | ---: |
| RF | 500 | 93 | 489 | 10 |
| Pair MLP | 500 | 294 | 316 | 7 |

Takeaway:

- This is the clearest saved evidence that Pair-MLP produces much more diverse rankings than RF.
- RF had near-collapse into the same pairs repeated across diseases.

### Vocab-pair ranking with low-sigma Pair-MLP

Artifact: `artifacts/vocab_repurpose/pairmlp_low_sigma_10d/summary.json`

Summary:

- Candidate drug count: `925`
- Unique pairs across exported rows: `294`
- Reused pair rows: `316`
- Pairs recommended for multiple diseases: `110`
- Mean diseases per pair: `1.7007`

Takeaway:

- Direct vocab-pair ranking preserved the Pair-MLP diversity advantage.

### MeDIC sequential Pair-MLP ranking

Artifact: `artifacts/model_compare_repurpose/pairmlp_topological512_medic_10d_seq/summary.json`

Summary:

- Unique pairs: `376`
- Reused pair rows: `203`
- Pairs recommended for multiple diseases: `79`

Takeaway:

- This was another strong diversity-oriented Pair-MLP ranking run over MeDIC candidates.

## Pair-MLP sigma sweep experiments

### Sigma sweep v2

Artifact: `artifacts/sigma_sweep_topological512_pairmlp_v2/sigma_sweep_report.json`

Sweep values:

- `5e-05`, `1e-04`, `5e-04`, `1e-03`, `5e-03`

Important points:

- Best held-out AUROC in this sweep: `0.9661` at `sigma=5e-04`
- Best held-out accuracy in this sweep: `0.9409` at `sigma=5e-03`
- Highest global uniqueness: `479` unique pairs at `sigma=5e-03`
- Lowest pair reuse: `42` reused pair rows at `sigma=5e-03`

Interpretation:

- Increasing sigma to `5e-03` improved ranking diversity dramatically while keeping held-out performance competitive.
- Very small sigma values tended to preserve better recall in some runs but gave more repeated pairs.

### Sigma sweep v3

Artifact: `artifacts/sigma_sweep_topological512_pairmlp_v3/sigma_sweep_report.json`

Sweep values:

- `1e-06`, `5e-06`, `1e-05`, `1e-02`, `5e-02`, `1e-01`

Important points:

- Best held-out AUROC in this sweep: `0.9685` at `sigma=1e-02`
- Best uniqueness: `493` unique pairs at `sigma=1e-02`
- Lowest reuse: `14` reused pair rows at `sigma=1e-02`
- Large sigma values `5e-02` and `1e-01` degraded held-out accuracy and recall

Interpretation:

- `sigma=1e-02` is the strongest saved single-run diversity setting in the repo.
- Too-large sigma eventually hurts classifier quality.

### Replicate sigma sweep

Artifact: `artifacts/sigma_sweep_topological512_pairmlp_replicates/sigma_replicates_report.json`

Setup:

- Sigmas: `1e-06` through `1e-01`
- Seeds: `13` through `22`
- Replicates per sigma: `10`

Key aggregate findings:

- Best mean held-out AUROC: `0.9685 ± 0.0011` at `sigma=1e-02`
- Best mean held-out accuracy: `0.9336 ± 0.0018` at `sigma=1e-06`
- Best diversity:
  - `488.0 ± 2.0` global unique pairs at `sigma=1e-02`
  - `22.7 ± 3.8` reused pair rows at `sigma=1e-02`
  - `58.02 ± 0.50` mean unique drugs in top-50 at `sigma=1e-02`

Interpretation:

- `sigma=1e-02` is the most defensible current Pair-MLP ranking setting when diversity matters.
- Very small sigmas are slightly safer on classification metrics, but they are much worse on ranking redundancy.

### Smoke replicate sweep

Artifact: `artifacts/sigma_replicates_smoke/sigma_replicates_report.json`

Compared:

- `sigma=1e-03`
- `sigma=1e-02`

Result:

- `sigma=1e-02` again dominated on uniqueness and had slightly better AUROC.
- `sigma=1e-03` had slightly better held-out accuracy/F1 in this tiny smoke comparison.

## Triplet-model experiments

Artifact family: `artifacts_triplet_lstm_full`

### Main triplet evaluation

Source:

- `artifacts_triplet_lstm_full/metrics_triplet.json`
- `artifacts_triplet_lstm_full/metrics_pair.json`
- `artifacts_triplet_lstm_full/val_metrics_best.json`

Validation best:

- Accuracy-oriented snapshot is not stored directly, but:
  - AUROC: `0.9916`
  - F1: `0.8932`
  - Precision: `0.8988`
  - Recall: `0.8876`
  - Balanced accuracy: `0.9382`

Triplet test metrics:

- Rows: `99690`
- AUROC: `0.9573`
- F1: `0.5855`
- Precision: `0.6257`
- Recall: `0.5501`
- Balanced accuracy: `0.7723`
- Enrichment factors:
  - EF5: `16.21`
  - EF10: `8.79`
  - EF20: `4.66`

Takeaway:

- The triplet model had strong ranking enrichment despite modest thresholded F1.

### Candidate-space triplet evaluation

Source:

- `artifacts_triplet_lstm_full/candidates/metrics_triplet.json`
- `artifacts_triplet_lstm_full/candidates/metrics_pair.json`

Candidate-space metrics:

| Metric | Triplet view | Pair-aggregated view |
| --- | ---: | ---: |
| AUROC | 0.7765 | 0.8081 |
| F1 | 0.0026 | 0.0091 |
| Precision | 0.0013 | 0.0046 |
| Recall | 0.5501 | 0.3599 |
| EF5 | 6.34 | 6.44 |
| EF10 | 4.54 | 4.64 |
| EF20 | 3.08 | 3.35 |

Takeaway:

- In a very large candidate space, thresholded classification metrics collapse, but enrichment remains meaningfully above random.
- This supports using the triplet model more as a ranking engine than as a hard classifier.

## Current practical conclusions

- For best held-out classification, the RF family is still the strongest saved baseline.
- For more diverse repurposing recommendations, Pair-MLP is better than RF and better than the saved LSTM ranking runs.
- For Pair-MLP ranking, `sigma=1e-02` is the current best saved tradeoff between diversity and performance.
- The refined GT increased specificity/precision but reduced sensitivity relative to the old GT.
- The triplet model is more compelling as a ranking model than as a thresholded classifier.
