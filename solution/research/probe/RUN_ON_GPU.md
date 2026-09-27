# Optional GPU network sweep

`nn_sweep.py` selects CUDA when PyTorch detects a GPU and otherwise runs on CPU. This is a research rerun guide, not a required setup for using the ready solution. The script expects the original analysis inputs described in `README.md`; they are not included in this branch.

From an authorized copy of the research workspace, activate an environment with PyTorch (CUDA build for GPU), NumPy, pandas and scikit-learn, then run the script from its expected analysis root so its relative input paths resolve. Example configuration syntax:

```powershell
python solution\research\probe\nn_sweep.py 256x2x0.01,512x2x0.01,256x3x0.01,1024x2x0.01 c16b1.0
```

The first argument is a comma-separated list of `WIDTHxDEPTHxWEIGHTDECAY` values. The second selects a stored probe variant such as `c4b0.5` or `c16b1.0`. The script prints one JSON result per configuration. It writes regenerable prediction arrays under `preds/`; those arrays are intentionally not versioned. Training cost depends on hardware and configuration; the saved reference for 256x2 was about 250 seconds on two CPU cores.

Do not commit a virtual environment or generated predictions. Confirm the first output reports `device: cuda` before interpreting a run as GPU-backed.
