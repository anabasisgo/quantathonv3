# Probe-assisted runtime model on `nick`

The model predicts simulation duration from a new QASM circuit and a threshold of 16, 64, or 512. It combines the existing Nick/Codex runtime estimate with an ExtraTrees estimate derived from a bounded matrix-product-state probe. Their durations are blended equally in log space. The existing learned timeout classifier runs first and returns the four-hour cap automatically when it predicts timeout; no customer timeout flag is required.

## Result and limits

The runnable one-second probe was evaluated on 1,497 rows from 532 circuits using the existing five circuit-held-out folds. All thresholds for a circuit were in one fold. Each probe tree was trained on the other four folds; the old model's saved out-of-fold predictions were used for the blend. Mean challenge score was **93.19%**, versus **92.19%** for the previous model on the same rows. The candidate caught **23/33** timeouts with **1/1,464** false alarms, made **24** errors greater than tenfold (previously 46), and landed **1,280/1,497** predictions within twofold. A whole-circuit paired bootstrap gave a +1.00 percentage-point gain, with a descriptive 95% interval of +0.56 to +1.46 points. The prior timeout classifier was tuned during development, and model choices were made after examining this corpus. These are research estimates, not an organizer hidden-set or new-customer result.

The probe accepts a practical QASM 2/3 subset. It uses a restricted arithmetic AST for angles; it never invokes Python `eval` on circuit text. Parsing uses a 20 MB input prefix, at most 400,000 emitted operations, a four-second parse budget, and a 4,096-qubit limit. MPS probing uses bond cap 16 and a one-second nominal budget. Probe work has a deadline 12 seconds after feature extraction starts; the pre-existing static parsers run before that deadline check. An automatic fallback for circuits with at least one million counted operations protects the harness latency cap. Unsupported gates, unresolved expressions, invalid probes, and extraction failures also fall back to the existing runtime model. Eleven training circuits took the deterministic fallback: nine for size and two because the probe could not be used. The full 532-circuit practice run returned all 1,596 requested predictions but found three parse times above 15 seconds. After the fallback change, the 15 slowest circuits were rerun; maximum parse time was 13.45 seconds, no harness warning occurred, and maximum prediction time was 0.012 seconds. A single MPS operation can overshoot the nominal time budget, so measure total harness latency on new workloads.

## Training data

- `../training_circuits/*.qasm.zst`: 532 compressed circuit files already tracked by Git LFS. Run `git lfs pull` after checkout if they appear as text pointers.
- `../runtime-data.csv`: 1,497 supplied labels, including timeout status and measured durations.
- `training_features.csv`: 532 rows of static, structural, and one-second safe-probe features extracted from the circuits. The model artifact records SHA-256 hashes of this file and the label CSV.

The fitted 300-tree artifact is `artifacts/probe-extra-trees.json.gz`. It was trained on the 1,467 labeled rows from 521 circuits eligible for the probe. Inference walks these trees with Python code and needs NumPy only for the probe; scikit-learn is a training dependency. The prior runtime and timeout artifacts remain in use.

## Run and retrain

From the repository root:

```bash
python -m pip install -r quantathon-harness/requirements.txt
python quantathon-harness/run.py --team "Your Team" --circuits training_circuits --out submission.csv
```

To regenerate the committed training features and refit the artifact, use Python 3.12 and:

```bash
python -m pip install -r quantathon-harness/requirements-training.txt
python quantathon-harness/build_probe_features.py --circuits training_circuits --out quantathon-harness/training_features.csv
python quantathon-harness/train_probe_model.py
```

Time-limited probe features can vary slightly with CPU speed, especially on large circuits. The committed feature table and artifact are the exact pair used for this model. Run `python -m unittest discover -s quantathon-harness -p test_probe_model.py` for the safety and new-circuit checks. The harness run on the training circuits is in-sample and must not be quoted as a generalization score.

The probe implementation is adapted from `origin/berni` commit `970d51a` (`solution/research/probe/probe.py`). Its unsafe angle evaluator has been replaced, and the data were re-extracted through the safer path. The Berni branch's 93.80% research blend used different folds and is not a direct head-to-head result.
