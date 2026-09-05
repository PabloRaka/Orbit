"""
End-to-End Integration Tests for Unified PhysLMEngine
=====================================================
"""

import numpy as np
import pytest
from src.engine import PhysLMEngine


@pytest.fixture
def engine():
    return PhysLMEngine(n_grid=1024, x_min=-20.0, x_max=20.0)


def test_unified_encode_decode_roundtrip(engine):
    """Verify that high-level API encodes and decodes accurately."""
    sample = "orbit"
    psi = engine.encode(sample)
    norm = engine.wave_engine.compute_norm(psi)
    assert np.isclose(norm, 1.0, atol=1e-5)

    recovered = engine.decode(psi, expected_length=len(sample))
    assert recovered == sample


def test_unified_wave_propagation(engine):
    """Verify that wave evolution through engine preserves unitary probability norm."""
    psi = engine.encode("ai")
    evolved = engine.propagate_wave(psi, steps=50, dt=0.002)
    norm = engine.wave_engine.compute_norm(evolved)
    assert np.isclose(norm, 1.0, atol=1e-10)


def test_unified_knowledge_completion(engine):
    """Verify end-to-end associative memory completion via high-level API."""
    engine.register_knowledge("sun:yellow")
    engine.register_knowledge("sea:blue")

    prompt = "sun:"
    target = "sun:yellow"
    completed, energy = engine.complete(prompt, target_length=len(target), steps=120)
    assert completed == target


def test_unified_syntax_validation(engine):
    """Verify high-level grammar checking via physical cavity."""
    valid_expr = "[{ ( < > ) }]"
    invalid_expr = "[{ ( < > } )]"

    is_valid_1, _ = engine.check_syntax(valid_expr)
    is_valid_2, _ = engine.check_syntax(invalid_expr)

    assert is_valid_1 is True
    assert is_valid_2 is False
