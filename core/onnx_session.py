"""
Shared ONNX Runtime session-configuration helpers.

Central place to resolve thread counts so a single key in system_config.yaml can
tune CPU utilisation for every ONNX session (detector + embedder) at once.

Motivation
----------
intra_op_num_threads caps how many cores a single ONNX call may use. Hardcoding it
to 4 means at most ~4 cores are active during inference (observed 433% CPU on a
32-vCPU box = 13.5% utilisation). Setting it to 0 lets ONNX Runtime use all
available cores (os.cpu_count()).
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import yaml
import onnxruntime as ort


_SYSTEM_CONFIG_PATH = "config/system_config.yaml"


def _load_system_config(system_config_path: str = _SYSTEM_CONFIG_PATH) -> dict:
    """Best-effort read of system_config.yaml; returns {} on any problem."""
    try:
        path = Path(system_config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [onnx_session] Could not read {system_config_path}: {e}")
    return {}


def resolve_thread_config(local_intra: int,
                          local_inter: int,
                          system_config_path: str = _SYSTEM_CONFIG_PATH
                          ) -> Tuple[int, int]:
    """
    Resolve effective (intra_op, inter_op) thread counts.

    Precedence:
        1. system_config.yaml -> onnx_threads  (central override, if present)
        2. the per-model yaml values passed in  (local_intra / local_inter)

    For the central key:
        onnx_threads.auto: true   -> (0, 0)   == use all cores
        onnx_threads.auto: false  -> (intra_op, inter_op) explicit values

    0 is ONNX Runtime's sentinel for "use all available cores".
    """
    intra, inter = local_intra, local_inter

    sys_cfg = _load_system_config(system_config_path)
    onnx_threads = sys_cfg.get('onnx_threads')
    if onnx_threads is not None:
        if onnx_threads.get('auto', True):
            intra, inter = 0, 0
        else:
            intra = onnx_threads.get('intra_op', 0)
            inter = onnx_threads.get('inter_op', 0)

    # Env override wins over everything. Lets a contended/oversubscribed host
    # (high CPU steal) cap threads WITHOUT editing the shared config, e.g.:
    #   ONNX_INTRA_OP=8 ONNX_INTER_OP=1 python ...
    # On a VM stealing ~half its cycles, fewer threads than cores avoids the
    # intra-op barrier thrashing that makes 32 threads slower than 8.
    env_intra = os.environ.get('ONNX_INTRA_OP')
    env_inter = os.environ.get('ONNX_INTER_OP')
    if env_intra is not None:
        try:
            intra = int(env_intra)
        except ValueError:
            print(f"  [onnx_threads] Ignoring invalid ONNX_INTRA_OP={env_intra!r}")
    if env_inter is not None:
        try:
            inter = int(env_inter)
        except ValueError:
            print(f"  [onnx_threads] Ignoring invalid ONNX_INTER_OP={env_inter!r}")

    return intra, inter


def configure_session_options(sess_options: ort.SessionOptions,
                              intra_op: int,
                              inter_op: int) -> None:
    """
    Apply resolved thread counts to a SessionOptions, enabling graph-node
    parallelism when running with all cores (intra_op == 0).
    """
    sess_options.intra_op_num_threads = intra_op
    sess_options.inter_op_num_threads = inter_op

    # Silence benign per-run "VerifyOutputSizes" shape-mismatch warnings. They fire
    # when a model has static output dims but we feed a dynamic input size (Phase 1c)
    # or a batch >1 (batched embeddings) — ORT still computes correct results. 3=ERROR.
    sess_options.log_severity_level = 3

    if intra_op == 0:
        # 0 == let ORT decide == os.cpu_count(); enable inter-op (graph node) parallelism.
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL


def effective_threads(intra_op: int) -> int:
    """Number of cores a session will actually use (0 sentinel -> os.cpu_count())."""
    return intra_op if intra_op else (os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Phase 1b — INT8 quantized model A/B switch
# ---------------------------------------------------------------------------
def resolve_model_path(original_path: str,
                       system_config_path: str = _SYSTEM_CONFIG_PATH) -> str:
    """
    Return the INT8 model path (`<stem>_int8.onnx`) when quantized models are
    enabled AND the file exists; otherwise the original FP32 path.

    Enabled by either:
      - system_config.yaml -> use_quantized_models: true
      - env USE_QUANTIZED_MODELS=1   (wins; lets you A/B without editing config)

    Default OFF — FP32 stays the production path until measured.
    """
    enabled = bool(_load_system_config(system_config_path).get('use_quantized_models', False))
    env = os.environ.get('USE_QUANTIZED_MODELS')
    if env is not None:
        enabled = env.strip().lower() in ('1', 'true', 'yes', 'on')

    if not enabled:
        return original_path

    p = Path(original_path)
    int8 = p.with_name(f"{p.stem}_int8{p.suffix}")
    if int8.exists():
        print(f"  [quant] using INT8 model: {int8.name}")
        return str(int8)
    print(f"  [quant] use_quantized_models ON but {int8.name} not found; "
          f"falling back to FP32 {p.name}")
    return original_path


# ---------------------------------------------------------------------------
# Phase 1d — CPU affinity (pin inference off the web stack)
# ---------------------------------------------------------------------------
def apply_cpu_affinity(system_config_path: str = _SYSTEM_CONFIG_PATH) -> Optional[set]:
    """
    Pin this process to a dedicated set of cores so ONNX threads stop fighting
    the co-located web stack (gunicorn/mysql/redis/celery). Reads:

        cpu_affinity:
          enabled: false
          cores: [0, 1, 2, 3, 4, 5, 6, 7]   # core ids to pin to

    Linux-only (os.sched_setaffinity). On other platforms it prints a taskset/
    cpuset hint and does nothing. Returns the applied core set, or None.

    Env override: CPU_AFFINITY="8-15" or "0,2,4,6" forces the core list.
    """
    cfg = _load_system_config(system_config_path).get('cpu_affinity', {}) or {}
    enabled = bool(cfg.get('enabled', False))
    cores = cfg.get('cores')

    env = os.environ.get('CPU_AFFINITY')
    if env:
        enabled = True
        cores = _parse_core_list(env)

    if not enabled:
        return None

    if not cores:
        print("  [affinity] enabled but no cores listed; skipping.")
        return None

    core_set = set(int(c) for c in cores)
    if not hasattr(os, 'sched_setaffinity'):
        print(f"  [affinity] not supported on this platform. On Linux this would pin to "
              f"{sorted(core_set)}. Workaround: launch with `taskset -c "
              f"{','.join(map(str, sorted(core_set)))} python ...`")
        return None

    try:
        avail = os.sched_getaffinity(0)
        core_set = {c for c in core_set if c < (os.cpu_count() or 1)}
        if not core_set:
            print("  [affinity] no valid cores after range check; skipping.")
            return None
        os.sched_setaffinity(0, core_set)
        print(f"  [affinity] pinned process to cores {sorted(core_set)} "
              f"(was {len(avail)} cores) — inference now isolated from the web stack.")
        return core_set
    except Exception as e:
        print(f"  [affinity] could not set affinity: {e}")
        return None


def _parse_core_list(spec: str):
    """Parse '0,2,4' or '8-15' or '0-3,8,10' into a list of ints."""
    out = []
    for part in spec.replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
