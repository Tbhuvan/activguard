# ActivGuard Probe — Autoresearch Program

Adapted from Karpathy's autoresearch for vulnerability detection probe optimisation.

## Context

You are autonomously improving a **linear probe on LLM residual stream activations** that
detects vulnerability signatures in code.  The probe is Layer 1 of ActivGuard — a
fast gate that reads the LLM's hidden states before tokens are emitted.

**Current baselines:**
- dolphin3:8b final-layer → AUC 0.6440
- CodeBERT layer-9       → AUC 0.9140  ← current best

## Setup

1. Read these files for full context:
   - `README.md` — repo context
   - `probe_autoresearch/prepare_probe.py` — fixed: data loading and evaluation. Do NOT modify.
   - `probe_autoresearch/train_probe_auto.py` — the file you modify.
2. Create branch: `git checkout -b autoresearch/probe-<tag>`
3. Initialise `results.tsv` with header: `commit\tval_auc\tstatus\tdescription`
4. Verify embeddings exist: `.activguard/embed_cache_dolphin3_8b.npz` and `.activguard/layer_cache/codebert_layers.npz`
5. Run baseline: `python probe_autoresearch/train_probe_auto.py`

## Experiment loop

LOOP FOREVER:

1. Check current git state
2. Modify `train_probe_auto.py` with an experimental idea
3. `git commit`
4. Run: `python probe_autoresearch/train_probe_auto.py > probe_run.log 2>&1`
5. Read metric: `grep "^val_auc:" probe_run.log`
6. If empty → crash. `tail -30 probe_run.log` to diagnose. Easy fix → fix and re-run. Hard → discard.
7. Log to `results.tsv` (do NOT commit this file)
8. If val_auc improved → keep commit (advance branch)
9. If val_auc same or worse → `git reset --hard HEAD~1`

**Goal: maximise val_auc (5-fold stratified CV ROC-AUC). Higher is better.**

## What you can change in train_probe_auto.py

Everything between the "EXPERIMENT" comment and the final three lines:

**Feature choices:**
- `data["X_dolphin"]` — 4096-dim dolphin3:8b final-layer (currently AUC 0.644)
- `data["X_codebert"]` — 768-dim CodeBERT layer-9 (currently AUC 0.914)
- `data["X_layers"][:, N, :]` — CodeBERT layer N (try layers 7, 8, 10, 11, 12)
- `np.hstack([...])` — concatenate multiple embedding sources
- PCA, feature selection, or other transformations

**Model choices (must have predict_proba):**
- `LogisticRegression(C=...)` — vary regularisation
- `SVC(probability=True, kernel='rbf', C=..., gamma=...)`
- `RandomForestClassifier(n_estimators=...)`
- `GradientBoostingClassifier(...)`
- `VotingClassifier(estimators=[...], voting='soft')` — ensemble
- `Pipeline([("scaler", ...), ("pca", PCA(n_components=...)), ("clf", ...)])` — dim reduction first

**Ideas to try (roughly ordered by expected impact):**
1. Try layer 8, 10, 11, 12 — layer 9 is best but margins are tight (0.902–0.914)
2. Concatenate layer 9 + layer 12 (last): captures both semantic and task-specific signal
3. SVM RBF kernel — often beats LogReg on moderate-dimensional features
4. Reduce dims with PCA(n_components=100) before SVM — may improve generalisation
5. Concatenate CodeBERT + dolphin3:8b — cross-model ensemble signal
6. GradientBoosting on top-K PCA components
7. VotingClassifier ensemble: LogReg + SVM + RF
8. Try different C values: 0.01, 0.1, 0.5, 2.0, 5.0, 10.0
9. Normalise per-class: subtract class mean, then probe (RepE-style)
10. Use difference vector: mean(vuln_embeddings) - mean(safe_embeddings) as probe direction

## What you CANNOT change

- `prepare_probe.py` — it is read-only
- The final three lines of `train_probe_auto.py` (evaluate_auc, print_summary calls)
- Do not install new packages

## Output format

The script prints:
```
---
val_auc:     0.914000
model_desc:  LogReg C=1.0 | CodeBERT layer-9
X_shape:  (100, 768)
---
```

## Results TSV format

```
commit  val_auc  status  description
a1b2c3d  0.914000  keep  baseline: LogReg C=1.0 CodeBERT L9
b2c3d4e  0.921000  keep  SVM RBF C=5.0 CodeBERT L9
c3d4e5f  0.910000  discard  PCA(50) + LogReg, underfit
```

## Simplicity criterion

Same as Karpathy's: prefer simpler models with equal or better AUC.
A 0.001 improvement from adding 50 lines of complexity: not worth it.
A 0.001 improvement from simplifying the pipeline: definitely keep.

## NEVER STOP

Once the experiment loop begins, do NOT pause. Run until manually interrupted.
