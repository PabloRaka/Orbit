"""
Project Resonon / PhysLM: Empirical Autoregressive Training & Generation Demo
=============================================================================
Demonstrates end-to-end physical language learning using Equilibrium Propagation
on continuous wave representations without backpropagation.
"""

import time
import numpy as np

from src.transducer import GaborWaveTransducer
from src.equilibrium_propagation import MemristiveCrossbarNetwork
from src.sequence_trainer import AutoregressiveSequenceTrainer


def run_autoregressive_demo():
    print("=" * 75)
    print("Project Resonon / PhysLM: Causal Next-Wave Autoregressive Training Demo")
    print("=" * 75)

    np.random.seed(42)

    # 1. Setup Physical Subsystems
    n_grid = 256
    dim_hid = 128
    transducer = GaborWaveTransducer(n_grid=n_grid, x_min=-10.0, x_max=10.0)
    network = MemristiveCrossbarNetwork(
        dim_in=n_grid,
        dim_hid=dim_hid,
        dim_out=n_grid,
        eta=0.04,
        beta=0.35,
        dt=0.15
    )
    trainer = AutoregressiveSequenceTrainer(
        transducer=transducer,
        network=network,
        context_window=1
    )

    corpus = "PHYSICS OF CONTINUOUS WAVES AND HARMONIC FIELDS"
    print(f"\n[1] Training Corpus ({len(corpus)} chars):")
    print(f"    \"{corpus}\"")

    transitions = trainer.dataset.extract_transitions(corpus)
    print(f"    Extracted {len(transitions)} causal (context, next_char) transitions.")

    # 2. Pre-Training Evaluation
    pre_gen = trainer.generate(seed="P", max_chars=12, temperature=0.0)
    pre_acc = trainer.evaluate_accuracy(transitions)
    print(f"\n[2] Pre-Training State:")
    print(f"    Accuracy on corpus transitions: {pre_acc * 100:.1f}%")
    print(f"    Untrained generation (seed='P'): \"{pre_gen}\"")

    # 3. Training Loop via Equilibrium Propagation
    print(f"\n[3] Executing Equilibrium Propagation Training (30 Epochs)...")
    print(f"    {'Epoch':<8}{'MSE Energy Loss':<20}{'Accuracy (%)':<15}{'Epoch Time (ms)':<15}")
    print("-" * 60)

    start_train = time.perf_counter()
    epochs = 30
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        loss = trainer.train_epoch(transitions, free_steps=25, nudge_steps=12)
        acc = trainer.evaluate_accuracy(transitions)
        t_ms = (time.perf_counter() - t0) * 1000.0

        if epoch in [1, 5, 10, 15, 20, 25, 30]:
            print(f"    {epoch:<8}{loss:<20.6f}{acc * 100:<15.1f}{t_ms:<15.2f}")

    total_time = time.perf_counter() - start_train
    post_acc = trainer.evaluate_accuracy(transitions)
    time_per_transition = (total_time / (epochs * len(transitions))) * 1000.0

    print("-" * 60)
    print(f"    Total training time: {total_time:.2f} s")
    print(f"    Latency per physical transition: {time_per_transition:.3f} ms")

    # 4. Post-Training Autoregressive Generation
    print(f"\n[4] Post-Training Autoregressive Generation (Greedy T=0.0):")
    for seed in ["P", "C", "W", "H"]:
        gen = trainer.generate(seed=seed, max_chars=14, temperature=0.0)
        print(f"    Seed '{seed}': \"{gen}\"")

    # 5. Boltzmann Thermal Noise Sampling (RFC-004)
    print(f"\n[5] Thermodynamic Boltzmann Sampling (RFC-004) at Various Temperatures:")
    seed = "P"
    for temp in [0.0, 0.2, 0.6, 1.2]:
        gen = trainer.generate(seed=seed, max_chars=14, temperature=temp)
        desc = "Ground state (deterministic)" if temp == 0.0 else f"Thermal noise T={temp}"
        print(f"    T={temp:<4} ({desc:<28}): \"{gen}\"")

    print("\n" + "=" * 75)
    print("Benchmark Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    run_autoregressive_demo()
