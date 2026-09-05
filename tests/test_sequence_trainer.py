"""
Unit and Integration Tests for Autoregressive Sequence Trainer & Generative Sampler
====================================================================================
"""

import numpy as np
import pytest

from src.transducer import GaborWaveTransducer
from src.equilibrium_propagation import MemristiveCrossbarNetwork
from src.sequence_trainer import CausalSequenceDataset, AutoregressiveSequenceTrainer
from src.engine import PhysLMEngine


def test_causal_sequence_dataset_extraction():
    """Verify sliding-window causal extraction from raw text."""
    transducer = GaborWaveTransducer(n_grid=128, x_min=-5.0, x_max=5.0)
    dataset = CausalSequenceDataset(transducer, context_window=2)

    text = "WAVE"
    transitions = dataset.extract_transitions(text)
    # Transitions: ("WA", "V"), ("AV", "E")
    assert len(transitions) == 2
    assert transitions[0] == ("WA", "V")
    assert transitions[1] == ("AV", "E")

    psi_ctx, psi_tgt = dataset.encode_transition("WA", "V")
    assert len(psi_ctx) == 128
    assert len(psi_tgt) == 128
    # Both states must be unitarily normalized
    assert np.isclose(np.sum(np.abs(psi_ctx) ** 2) * transducer.dx, 1.0, atol=1e-3)
    assert np.isclose(np.sum(np.abs(psi_tgt) ** 2) * transducer.dx, 1.0, atol=1e-3)


def test_vectorized_basis_probe_measurement():
    """Verify that vectorized probe measurement achieves maximum overlap on ground truth."""
    transducer = GaborWaveTransducer(n_grid=128, x_min=-5.0, x_max=5.0)
    trainer = AutoregressiveSequenceTrainer(transducer=transducer, dim_hid=32, context_window=1)

    for char in ["A", "X", "7", " "]:
        psi = transducer.encode(char)
        measured_char, overlaps = trainer.measure_wave(psi, temperature=0.0)
        assert measured_char == char
        # Self-overlap must be maximum
        best_idx = np.argmax(overlaps)
        assert transducer.charset[best_idx] == char


def test_equilibrium_propagation_sequence_learning():
    """Verify that EqProp minimizes contrastive energy loss on text transitions."""
    np.random.seed(42)
    transducer = GaborWaveTransducer(n_grid=128, x_min=-5.0, x_max=5.0)
    network = MemristiveCrossbarNetwork(
        dim_in=128,
        dim_hid=64,
        dim_out=128,
        eta=0.04,
        beta=0.35,
        dt=0.15
    )
    trainer = AutoregressiveSequenceTrainer(
        transducer=transducer,
        network=network,
        context_window=1
    )

    corpus = "PHYSICS"
    result = trainer.train(corpus, epochs=15, free_steps=20, nudge_steps=10)

    # 1. Energy loss must decrease through learning
    assert result["final_loss"] < result["initial_loss"], (
        f"Loss failed to decrease: init={result['initial_loss']:.4f}, final={result['final_loss']:.4f}"
    )
    # 2. History must contain exactly 15 epoch entries
    assert len(result["history"]) == 15


def test_autoregressive_generation_deterministic_vs_boltzmann():
    """Verify deterministic generation at T=0 and thermodynamic sampling at T>0."""
    np.random.seed(123)
    transducer = GaborWaveTransducer(n_grid=128, x_min=-5.0, x_max=5.0)
    trainer = AutoregressiveSequenceTrainer(transducer=transducer, dim_hid=32, context_window=1)

    # Deterministic generation at T=0
    gen1 = trainer.generate(seed="P", max_chars=8, temperature=0.0)
    gen2 = trainer.generate(seed="P", max_chars=8, temperature=0.0)
    assert gen1 == gen2
    assert len(gen1) == 9  # 1 seed + 8 generated

    # Stop chars behavior
    stop_gen = trainer.generate(seed="P", max_chars=8, temperature=0.0, stop_chars=[gen1[1]])
    assert len(stop_gen) == 1  # Immediately stopped at first char


def test_physlm_engine_autoregressive_integration():
    """Verify that PhysLMEngine provides a seamless high-level sequence learning API."""
    np.random.seed(42)
    engine = PhysLMEngine(n_grid=128, x_min=-5.0, x_max=5.0, context_window=1)

    corpus = "ATOM"
    res = engine.train_autoregressive(corpus, epochs=5, free_steps=15, nudge_steps=8)
    assert res["total_epochs"] == 5
    assert res["transitions_count"] == 3

    out = engine.generate_autoregressive(seed="A", max_chars=3, temperature=0.0)
    assert len(out) == 4
    assert out.startswith("A")


def test_physical_crossbar_layer_held_out_evaluation():
    """Verify that PhysicalCrossbarLayer learns natural transitions with positive margin on held-out transitions."""
    np.random.seed(42)
    transducer = GaborWaveTransducer(n_grid=128, x_min=-5.0, x_max=5.0)
    trainer = AutoregressiveSequenceTrainer(
        transducer=transducer,
        context_window=3,
        layer_type="direct"
    )

    train_corpus = "CAT IS FAST. DOG IS SMALL. "
    held_out_corpus = "CAT IS SMALL. DOG IS FAST. "
    train_trans = trainer.dataset.extract_transitions(train_corpus)
    held_trans = trainer.dataset.extract_transitions(held_out_corpus)

    for _ in range(80):
        trainer.train_epoch(train_trans)

    res_train = trainer.evaluate_transitions(train_trans)
    res_held = trainer.evaluate_transitions(held_trans)

    assert res_train["mean_margin"] > 0, f"Train margin not positive: {res_train['mean_margin']}"
    assert res_held["mean_margin"] > 0, f"Held-out margin not positive: {res_held['mean_margin']}"
    assert res_train["accuracy"] > 0.80


def test_rollout_with_metrics_modes_and_horizons():
    """Verify rollout_with_metrics works for both Projective Restoration and Free-Flight modes."""
    np.random.seed(42)
    transducer = GaborWaveTransducer(n_grid=128, x_min=-5.0, x_max=5.0)
    trainer = AutoregressiveSequenceTrainer(
        transducer=transducer,
        context_window=2,
        layer_type="direct"
    )

    corpus = "ABCDEFGH"
    trans = trainer.dataset.extract_transitions(corpus)
    for _ in range(40):
        trainer.train_epoch(trans)

    # Test Projective mode
    res_proj = trainer.rollout_with_metrics(seed="AB", horizon=4, mode="projective", reference_seq=corpus)
    assert res_proj["horizon"] == 4
    assert "L_H" in res_proj
    assert "VCR" in res_proj
    assert len(res_proj["trajectory_sq_errors"]) == 4

    # Test Free-Flight mode
    res_ff = trainer.rollout_with_metrics(seed="AB", horizon=4, mode="free_flight", reference_seq=corpus)
    assert res_ff["horizon"] == 4
    assert "L_H" in res_ff
    assert "mean_delta_drift" in res_ff

