"""All tunable constants of the NN A submission pipeline in one place."""

CAP_SECONDS = 14_400.0             # 4-hour timeout cap; predictions are not clipped, >= cap signals a timeout
THRESHOLDS = (16, 64, 512)

# --- time budgets (seconds). featurize must stay well under the harness's 15 s cap.
EXTRACTOR_BUDGET_S = 6.0           # team v3 structural extractor (compact features)
PROBE_SKIP_AFTER_S = 9.0           # watchdog: if featurize has already used this much, skip the probe (features -> missing)

# --- probe parser limits (deterministic: independent of machine speed)
PROBE_MAX_CHARS = 2_000_000        # parse at most this prefix of the QASM text
PROBE_MAX_OPS = 150_000            # and at most this many expanded 1q/2q operations

# --- truncated MPS probe
PROBE_BOND_CAP = 16
# Deterministic work budget in "reference seconds": work = FLOP_COST*flops + OP_COST*ops + SVD_COST*svds.
# Calibrated so that 1.0 unit ~ 1 s on the development machine (2-core cloud VM).
PROBE_WORK_BUDGET = 1.0
FLOP_COST, OP_COST, SVD_COST = 9.6e-9, 1.3e-5, 6.0e-5
PROBE_SAFETY_WALL_S = 3.0          # hard wall-clock stop in case a machine is much slower than the reference

# --- feature order used by the network (44 features; 4 threshold inputs are appended by network.py)
COMPACT_FEATURES = [
    'n_qubits', 'unitary_count', 'twoq_count', 'multiq_count', 'gate_depth', 'twoq_depth', 'measurement_count',
    'reset_count', 'twoq_unique_edges', 'twoq_density', 'twoq_degree_max', 'twoq_components', 'twoq_reuse',
    'twoq_top_pair_share', 'twoq_pair_mass_entropy', 'twoq_native_max_cut', 'twoq_native_mean_cut',
    'twoq_native_mean_span', 'twoq_rcm_max_cut', 'twoq_rcm_mean_cut', 'twoq_rcm_mean_span',
    'generic_critical_depth_twoq', 'generic_entanglement_ratio_twoq', 'generic_liveness', 'generic_parallelism',
    'generic_program_communication_twoq']
BOUND_FEATURES = [
    'span_mean', 'span_max', 'bound_max_log2', 'bound_mean_log2', 'struct_logcost_16', 'struct_logcost_64',
    'struct_logcost_512', 'struct_frac_bonds_ge_16', 'struct_frac_bonds_ge_64', 'struct_frac_bonds_ge_512']
PROBE_FEATURES = [
    'probe_frac_bonds_saturated', 'probe_mean_log2_bond', 'probe_max_entropy', 'probe_log_discarded',
    'probe_trunc_frac', 'probe_first_sat_frac', 'probe_log_flops', 'probe_frac_ops_done']
FEATURES = COMPACT_FEATURES + BOUND_FEATURES + PROBE_FEATURES
