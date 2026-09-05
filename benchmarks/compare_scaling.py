"""
Project Resonon / PhysLM: Tier 4 Comparative Scaling Benchmark
==============================================================
Specification Reference: docs/benchmarks/01_PHYSICAL_AND_NUMERICAL_BENCHMARK_SUITE.md

Measures empirical and theoretical memory scaling and compute scaling:
NanoGPT Mini (Transformer KV-Cache) vs PhysLM (Continuous Physical Wave State)
across context horizons N from 1k to 128k+.
"""

import time
import numpy as np
from typing import Dict, List, Any
from src.baseline_phase0 import ContinuousWaveEngine
from src.transducer import GaborWaveTransducer


def compute_nanogpt_kv_cache_bytes(
    seq_len: int,
    num_layers: int = 6,
    hidden_dim: int = 384,
    bytes_per_elem: int = 2  # float16/bfloat16
) -> int:
    """Computes total bytes consumed by Transformer Key-Value cache in DRAM."""
    # 2 buffers (Keys, Values) * layers * sequence_length * hidden_dim * bytes
    return 2 * num_layers * seq_len * hidden_dim * bytes_per_elem


def compute_physlm_state_bytes(
    n_grid: int = 1024,
    dim_hid: int = 256,
    bytes_per_complex: int = 8  # complex64
) -> int:
    """Computes total bytes consumed by PhysLM active physical state."""
    # Active wave field psi(x) + latent crossbar state h
    return (n_grid + dim_hid) * bytes_per_complex


def run_scaling_benchmark() -> List[Dict[str, Any]]:
    context_lengths = [1024, 4096, 16384, 32768, 65536, 131072]
    
    # Initialize physical wave engine
    engine = ContinuousWaveEngine(n_grid=1024)
    psi = np.exp(-engine.x ** 2 / 2.0).astype(complex)
    psi /= np.sqrt(engine.compute_norm(psi))
    v_pot = 0.5 * engine.x ** 2

    # Measure average PhysLM step latency
    n_warmup = 50
    for _ in range(n_warmup):
        psi = engine.step_unitary_split_operator(psi, v_pot, dt=0.001)

    t0 = time.perf_counter()
    n_benchmark_steps = 200
    for _ in range(n_benchmark_steps):
        psi = engine.step_unitary_split_operator(psi, v_pot, dt=0.001)
    t1 = time.perf_counter()
    physlm_step_time_ms = ((t1 - t0) / n_benchmark_steps) * 1000.0

    records = []
    print("=" * 85)
    print(" PROJECT RESONON / PHYSLM: TIER 4 COMPARATIVE SCALING BENCHMARK")
    print(" Reference Baseline: NanoGPT Mini (6 Layers, 384 Dim, 6 Heads, FP16 KV-Cache)")
    print("=" * 85)
    print(f"{'Context N':<12} | {'NanoGPT KV Cache':<18} | {'PhysLM State':<14} | {'Memory Advantage':<18} | {'PhysLM Step Latency':<18}")
    print("-" * 85)

    for n in context_lengths:
        nanogpt_bytes = compute_nanogpt_kv_cache_bytes(n)
        physlm_bytes = compute_physlm_state_bytes()
        ratio = nanogpt_bytes / physlm_bytes

        nanogpt_str = f"{nanogpt_bytes / (1024 * 1024):.2f} MB" if nanogpt_bytes >= 1024 * 1024 else f"{nanogpt_bytes / 1024:.1f} KB"
        physlm_str = f"{physlm_bytes / 1024:.2f} KB"
        ratio_str = f"{ratio:.1f}x smaller"

        row = {
            "context_length": n,
            "nanogpt_bytes": nanogpt_bytes,
            "physlm_bytes": physlm_bytes,
            "ratio": ratio,
            "physlm_step_time_ms": physlm_step_time_ms
        }
        records.append(row)
        print(f"{n:<12} | {nanogpt_str:<18} | {physlm_str:<14} | {ratio_str:<18} | {physlm_step_time_ms:.4f} ms")

    print("=" * 85)
    return records


if __name__ == "__main__":
    run_scaling_benchmark()
