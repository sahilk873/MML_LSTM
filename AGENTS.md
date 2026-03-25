# Agent Instructions

This repository keeps an experiment memory in `EXPERIMENTS.md`.

Any coding agent working in this repo must follow these rules:

1. If you run, modify, or analyze an experiment and it produces a result worth keeping, update `EXPERIMENTS.md` in the same task.
2. Do not leave important experiment outcomes only inside artifact directories, terminal output, or chat history.
3. Record:
   - experiment name or artifact path
   - what changed from prior runs
   - the key metrics
   - the practical takeaway
4. If results are preliminary, contradictory, or incomplete, say so explicitly in `EXPERIMENTS.md`.
5. Do not delete old experiment entries just because a newer run exists. Mark newer runs as superseding older ones when appropriate.
6. When an experiment changes the current best-known setting or baseline, update the conclusion section in `EXPERIMENTS.md`.
7. Prefer reading saved metrics files such as `metrics.json`, `summary.json`, `metrics_summary.json`, `val_metrics_best.json`, or equivalent outputs rather than relying on memory.

Operational default:

- Before finishing a task that includes an experiment run or experiment analysis, check whether `EXPERIMENTS.md` needs an update.
