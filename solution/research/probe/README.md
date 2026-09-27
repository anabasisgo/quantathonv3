# Probe features and model-family experiments

Exploratory Quantum Rings research snapshot (2026-09-26). These experiments evaluate bounded circuit probes as extra runtime-prediction features. They do not modify or replace the ready v5 harness model, and none of the reported scores is an organizer hidden-set result.

## Main findings

- On the saved grouped five-fold split, adding the original cap-16, 3-second probe features to compact static features improved the official mean score for all three tested model families: ExtraTrees 89.05% to 92.32%, SVR 90.63% to 93.30%, and MLP 91.40% to 93.60%. A paired whole-circuit bootstrap estimated gains of +3.27 [+2.27, +4.26], +2.67 [+1.73, +3.59], and +2.20 [+1.47, +2.96] percentage points, respectively.
- The gain is split-sensitive. For compact+probe MLP, the saved stress scores were 90.21% on larger circuits, 79.97% on held-out QASM3, 76.40% on held-out other QASM2, and 81.68% on held-out Sycamore-like QASM2. These stress tests are separate populations and should not be averaged into a single expected score.
- Cheaper probes captured much of the benefit: cap 4 / 0.5 s added +2.38 points for SVR and +2.83 for ExtraTrees over their compact-feature baselines in the variant comparison. Cap 16 / 1 s was comparable to cap 16 / 3 s for SVR; cap 32 did not help. The structural bound without simulation added approximately zero for ExtraTrees and about +0.8 for SVR.
- In the saved neural-network size sweep, a two-layer width-256 MLP scored 94.21%, compared with 93.65% at width 128; paired gain +0.56 [+0.29, +0.86]. This is exploratory model selection on the same research split, not an independent confirmation or a promotion decision.
- Compact static features alone predicted a >=10x runtime blow-up between thresholds 16 and 512 with AUC 0.999 in the saved experiment. Restricting probing only to predicted blow-ups reduced score by 1–2 points, suggesting probe information also helps on circuits without that blow-up.
- The best recorded compact+probe blend scored 93.80% in the grouped CV table. This experiment used no timeout gate, so its figures are not directly comparable to the locked v5 result (90.10% on four development folds with a gate) or the v5 internal holdout (89.88%).

Scores use the challenge formula, with evaluation rows grouped by circuit and all preprocessing/model tuning inside training folds. The ordinary grouped CV and source/size stress results are selection-affected research estimates. See `model_comparison.csv`, `metrics_table.csv`, `svm_vs_trees.csv`, and `round2_results.jsonl` for recorded results and metrics.

## Probe design and limits

`probe.py` parses a practical QASM 2/3 subset into a gate stream and computes (1) a per-cut structural upper bound on log2 bond dimension and (2) truncated matrix-product-state (MPS) features under a bond cap and time budget. `probe16.jsonl` holds the original cap-16, 3-second run for 532 circuits. `probe_variants.jsonl` holds six cap/budget configurations, with prefixed feature names. Each JSONL row corresponds to one circuit.

The small-circuit check in `test_probe.py` compared the simulator against exact statevectors to approximately 1e-15 on 330 random circuits. This validates those tested cases, not all parser/gate combinations. The probe is a research prototype: it uses Python `eval` for angle expressions and approximates some undefined many-qubit gates. Do not deploy it on untrusted QASM or use its features as a production parser without replacing expression evaluation with a restricted parser and reviewing unsupported gate semantics. Feature extraction is bounded, may stop partway through a circuit, and reports coverage; the original run used a 4 s parse budget, 3 s probe budget and 20 MB text prefix.

## Files

- `probe.py`, `test_probe.py`: parser, structural bound, truncated MPS and focused exactness check.
- `run_probe.py`, `run_variants.py`: bounded feature-generation scripts. They expect circuit files in a local `qasm/` directory and are not turnkey without the challenge circuits.
- `probe16.jsonl`, `probe_variants.jsonl`: per-circuit feature data for 532 circuits.
- `models_test.py`, `eval_variants.py`, `nn_sweep.py`, `probe_importance.py`, `analyze.py`, `metrics.py`, `explode_test.py`, `adaptive_test.py`, `gate_test.py`, `svm_test.py`: model, ablation, diagnostic and robustness experiments.
- `model_comparison.csv`, `metrics_table.csv`, `svm_vs_trees.csv`, `round2_results.jsonl`: compact result tables/logs.
- `RUN_ON_GPU.md`: optional neural-network sweep notes. Prediction arrays and environments are reproducible outputs and are not included.

Some analysis scripts read the original local research tables (`data_first/...`, `modeling_v5/FEATURE_SETS.json`, and `data_first/feature_v4/inference_features_v4.csv`) that are not part of this snapshot. Recreate those paths from the authorized research workspace before rerunning model comparisons. The bundled challenge archive, `preds/`, virtual environments and image files are intentionally absent.
