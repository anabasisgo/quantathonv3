"""Focused safety and new-circuit inference checks."""
import math
import unittest

import probe_runtime
from model import RuntimeModel

SIMPLE = """OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
h q[0];
cx q[0],q[1];
"""


class ProbeModelTests(unittest.TestCase):
    def test_restricted_angles(self):
        self.assertAlmostEqual(probe_runtime._eval("sin(pi/2)+2^3", {}), 9.0)
        self.assertAlmostEqual(probe_runtime._eval("theta/2", {"theta": 1.2}), 0.6)
        with self.assertRaises(ValueError):
            probe_runtime._eval("__import__('os').system('echo forbidden')", {})

    def test_unknown_gate_falls_back(self):
        unknown = SIMPLE + "mystery q[0],q[1];\n"
        self.assertFalse(probe_runtime.featurize_text(unknown).get("usable"))

    def test_new_circuit_prediction(self):
        model = RuntimeModel()
        features = model.featurize(SIMPLE)
        self.assertTrue(features["probe"].get("usable"))
        for threshold in (16, 64, 512):
            prediction = model.predict(features, threshold)
            self.assertTrue(math.isfinite(prediction))
            self.assertGreater(prediction, 0.0)
            self.assertLessEqual(prediction, 14400.0)


if __name__ == "__main__":
    unittest.main()
