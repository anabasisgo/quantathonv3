"""Extract the safe probe and static features from compressed training circuits."""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MODEL_DIR = Path(os.environ.get("QUANTATHON_MODEL_DIR", Path(__file__).resolve().parent))
sys.path.insert(0, str(MODEL_DIR))
import zstandard as zstd
from model import featurize as static_featurize
from probe_runtime import featurize_text

STATIC = (
    "qubits", "ops", "depth", "oneq", "twoq", "multiq", "cx", "swap",
    "measure", "reset", "custom_calls", "control", "twoq_density", "log_qasm_chars",
    "name_ccx", "name_cp", "name_rx", "name_ry", "name_rz", "name_u", "name_u1",
    "name_u2", "name_u3", "name_h", "name_x", "name_t", "name_tdg", "name_cz",
    "name_swap", "name_reset", "name_barrier", "qasm3",
)
STRUCTURAL = (
    "span_mean", "span_max", "bound_max_log2", "bound_mean_log2",
    "struct_logcost_16", "struct_frac_bonds_ge_16", "struct_logcost_64",
    "struct_frac_bonds_ge_64", "struct_logcost_512", "struct_frac_bonds_ge_512",
)
PROBE = (
    "probe_ok", "probe_frac_ops_done", "probe_twoq_done",
    "probe_frac_bonds_saturated", "probe_mean_log2_bond", "probe_max_entropy",
    "probe_discarded", "probe_log_discarded", "probe_trunc_frac",
    "probe_first_sat_frac", "probe_log_flops",
)


def extract(path_string):
    path = Path(path_string)
    with path.open("rb") as stream:
        text = zstd.ZstdDecompressor().stream_reader(stream).read().decode("utf-8", "replace")
    static = static_featurize(text)
    try:
        probe = featurize_text(text, chi=16, parse_budget=4.0, probe_budget=1.0)
    except Exception as exc:
        probe = {"usable": False, "reason": type(exc).__name__}
    row = {"filename": path.name[:-4], "probe_usable": int(bool(probe.get("usable")))}
    row.update({key: static.get(key, 0.0) for key in STATIC})
    row.update({key: probe.get(key, 0.0) for key in STRUCTURAL + PROBE})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuits", type=Path, default=Path("training_circuits"))
    parser.add_argument("--out", type=Path, default=Path("quantathon-harness/training_features.csv"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    paths = sorted(args.circuits.glob("*.qasm.zst"))
    if not paths:
        raise SystemExit("No compressed circuits found")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract, str(path)): path.name for path in paths}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 25 == 0 or len(rows) == len(paths):
                print(f"extracted {len(rows)}/{len(paths)}", flush=True)
    rows.sort(key=lambda row: row["filename"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filename", "probe_usable") + STATIC + STRUCTURAL + PROBE)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
