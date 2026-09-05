"""
Unit and Integration Tests for Continuous Associative Memory & Completion
==========================================================================
"""

import numpy as np
import pytest
from src.transducer import GaborWaveTransducer
from src.associative_memory import ContinuousAssociativeMemory


@pytest.fixture
def memory_system():
    transducer = GaborWaveTransducer(n_grid=1024, x_min=-20.0, x_max=20.0)
    memory = ContinuousAssociativeMemory(transducer, beta=15.0, alpha_clamp=0.8, dt=0.08)
    
    # Register concept pairs
    memory.store("cat:meow")
    memory.store("dog:bark")
    memory.store("sky:blue")
    memory.store("fire:hot")
    return memory


def test_pattern_storage_and_orthogonal_separation(memory_system):
    """Verify stored patterns are registered and retain distinct energy wells."""
    assert len(memory_system.patterns) == 4
    
    # Overlap of pattern with itself is 1.0, cross-overlap between distinct concepts is low
    p0 = memory_system.patterns[0]
    p1 = memory_system.patterns[1]
    
    self_overlap = memory_system.compute_overlap(p0, p0)
    cross_overlap = memory_system.compute_overlap(p0, p1)
    
    assert np.isclose(self_overlap, 1.0, atol=1e-4)
    assert cross_overlap < 0.35, f"Cross-talk too high: {cross_overlap}"


def test_language_completion_cat(memory_system):
    """Verify that incomplete prompt 'cat:' settles into 'cat:meow'."""
    prompt = "cat:"
    target = "cat:meow"
    
    completed_text, energy = memory_system.complete(
        prompt_text=prompt,
        total_expected_length=len(target),
        steps=120
    )
    assert completed_text == target, f"Expected '{target}', but got '{completed_text}'"


def test_language_completion_dog(memory_system):
    """Verify that incomplete prompt 'dog:' settles into 'dog:bark'."""
    prompt = "dog:"
    target = "dog:bark"
    
    completed_text, energy = memory_system.complete(
        prompt_text=prompt,
        total_expected_length=len(target),
        steps=120
    )
    assert completed_text == target, f"Expected '{target}', but got '{completed_text}'"


def test_completion_under_thermal_noise(memory_system):
    """Verify that associative infilling converges accurately even under analog thermal noise."""
    prompt = "sky:"
    target = "sky:blue"
    
    np.random.seed(42)
    completed_text, energy = memory_system.complete(
        prompt_text=prompt,
        total_expected_length=len(target),
        steps=150,
        noise_sigma=0.02
    )
    assert completed_text == target, f"Noise infilling failed: expected '{target}', got '{completed_text}'"


def test_truncation_generalization(memory_system):
    """Verify that a heavily truncated cue ('ca') still pulls into 'cat:meow'."""
    prompt = "ca"
    target = "cat:meow"
    completed_text, _ = memory_system.complete(
        prompt_text=prompt,
        total_expected_length=len(target),
        steps=120
    )
    assert completed_text == target, f"Truncation failed: expected '{target}', got '{completed_text}'"


def test_near_neighbor_suffix_competition():
    """Verify positive margin when two concepts share an identical suffix."""
    transducer = GaborWaveTransducer(n_grid=512, x_min=-15.0, x_max=15.0)
    memory = ContinuousAssociativeMemory(transducer, beta=15.0, alpha_clamp=0.8, dt=0.04)
    memory.store("kucing:lucu")
    memory.store("anjing:lucu")

    completed, _ = memory.complete("kucing:", total_expected_length=11, steps=120)
    assert completed == "kucing:lucu"

    psi_rel = transducer.encode(completed)
    s_tgt = memory.compute_overlap(memory.patterns[0], psi_rel)
    s_comp = memory.compute_overlap(memory.patterns[1], psi_rel)
    margin = s_tgt - s_comp
    assert margin > 0.15, f"Near-neighbor margin too small: {margin}"


def test_gated_dyck_coupling_infilling():
    """Verify semantic infilling inside Dyck formal grammar brackets."""
    from src.dyck_resonator import PhaseLockingDyckCavity
    cavity = PhaseLockingDyckCavity(max_depth=16)
    transducer = GaborWaveTransducer(n_grid=512, x_min=-15.0, x_max=15.0)
    memory = ContinuousAssociativeMemory(transducer, beta=15.0, alpha_clamp=0.8, dt=0.04)
    memory.store("api:panas")

    raw_completed, _ = memory.complete("api:", total_expected_length=9, steps=120)
    structured_expr = f"([{raw_completed}])"
    is_valid, tel = cavity.parse(structured_expr)

    assert is_valid is True
    assert np.isclose(tel["residual_energy"], 0.0, atol=1e-5)

