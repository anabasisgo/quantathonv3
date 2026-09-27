"""
model.py  --  harness entry point for the NN A submission.

  featurize(qasm_text)       -> 44 features: compact circuit structure, a structural bond-dimension bound,
                                and a truncated matrix-product-state probe (see nnrt/).
  predict(features, thr)     -> seconds, from a 5-seed 1024-512-256 MLP ensemble trained on log10 runtime.
                                Predictions >= 14400 s signal an expected timeout.
Retrain with the scripts in ../training (build_features.py -> cross_validate.py -> train.py).
"""
from pathlib import Path

import nnrt

CAP_SECONDS = 4 * 60 * 60


class RuntimeModel:
    """Harness adapter for feature extraction and artifact-backed runtime inference."""

    def __init__(self, artifacts_dir=None):
        """Load the trained ensemble from the supplied or default artifact directory."""
        self.network = nnrt.RuntimeNetwork(artifacts_dir or Path(__file__).resolve().parent / "artifacts")

    def featurize(self, qasm_text: str) -> dict:
        """Return model features and stage timings for one decompressed QASM circuit."""
        return nnrt.featurize(qasm_text)

    def predict(self, features: dict, threshold: int) -> float:
        """Predict runtime in seconds for one circuit and bond threshold."""
        return float(10.0 ** self.network.predict_log10(features, int(threshold)))
