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
from typing import Tuple

import yaml
import onnxruntime as ort


_SYSTEM_CONFIG_PATH = "config/system_config.yaml"


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

    try:
        path = Path(system_config_path)
        if path.exists():
            with open(path, 'r') as f:
                sys_cfg = yaml.safe_load(f) or {}
            onnx_threads = sys_cfg.get('onnx_threads')
            if onnx_threads is not None:
                if onnx_threads.get('auto', True):
                    intra, inter = 0, 0
                else:
                    intra = onnx_threads.get('intra_op', 0)
                    inter = onnx_threads.get('inter_op', 0)
    except Exception as e:
        # Never let config issues break session creation; fall back to local values.
        print(f"  [onnx_threads] Could not read {system_config_path}, using local config: {e}")

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

    if intra_op == 0:
        # 0 == let ORT decide == os.cpu_count(); enable inter-op (graph node) parallelism.
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL


def effective_threads(intra_op: int) -> int:
    """Number of cores a session will actually use (0 sentinel -> os.cpu_count())."""
    return intra_op if intra_op else (os.cpu_count() or 1)
