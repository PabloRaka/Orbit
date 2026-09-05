"""
Unit and Benchmark Tests for Phase-Locking Dyck Cavity Resonator
================================================================
Benchmark Tier 3, Stage 2: Formal Grammar Resonance without a Digital Stack.
"""

import pytest
from src.dyck_resonator import PhaseLockingDyckCavity


@pytest.fixture
def cavity():
    return PhaseLockingDyckCavity(max_depth=32)


def test_basic_bracket_types(cavity):
    """Verify ground state convergence for single pairs of each bracket type."""
    for pair in ["()", "[]", "{}", "<>"]:
        valid, info = cavity.parse(pair)
        assert valid is True, f"Failed on simple pair '{pair}'"
        assert info["final_excitation"] == 0
        assert info["residual_energy"] < 1e-10


def test_interleaved_valid_structures(cavity):
    """Verify adjacent and combined balanced grammar sequences."""
    test_cases = [
        "()[]{}<>",
        "[()]{}<[]>",
        "({[]}) <{()}> [[]]",
        "def foo(x): return [y for y in {1, 2}]"
    ]
    for expr in test_cases:
        valid, info = cavity.parse(expr)
        assert valid is True, f"Failed on valid interleaved '{expr}'"


def test_recursive_nesting_up_to_depth_16(cavity):
    """
    Tier 3, Stage 2 Target: 100% syntax validity on recursive depths up to D = 16.
    """
    open_chars = ["(", "[", "{", "<"]
    close_chars = [")", "]", "}", ">"]

    for depth in range(1, 17):
        # Generate alternating nested structure
        prefix = "".join(open_chars[d % 4] for d in range(depth))
        suffix = "".join(close_chars[d % 4] for d in reversed(range(depth)))
        expr = prefix + suffix

        valid, info = cavity.parse(expr)
        assert valid is True, f"Failed at depth {depth}: {expr}"
        assert info["max_depth"] == depth
        assert info["final_excitation"] == 0
        assert info["residual_energy"] < 1e-10


def test_lifo_order_violations(cavity):
    """
    Crucial Test: Cross-bracket order violations must trigger non-zero residual phase anomaly.
    """
    invalid_cases = [
        "[(])",       # Classic crossing violation
        "([)]",       # Classic crossing violation
        "{[}]",
        "<{( )]>",
        "[[({ )]}]]", # Deep crossing violation
        "{[ ( ] )}"
    ]
    for expr in invalid_cases:
        valid, info = cavity.parse(expr)
        assert valid is False, f"Should have failed LIFO check on '{expr}'"
        assert info["residual_energy"] > 0.05, f"Expected anomaly energy on '{expr}', got {info['residual_energy']}"


def test_balance_violations(cavity):
    """Verify detection of unbalanced brackets (too many opens or closes)."""
    unbalanced_cases = [
        "((()",     # Missing closes
        "())",      # Excess close
        "]",        # Leading close
        "{{[()]",   # Unclosed outer braces
        "(()))",    # Trailing extra close
        "><"        # Reversed
    ]
    for expr in unbalanced_cases:
        valid, info = cavity.parse(expr)
        assert valid is False, f"Should have failed balance check on '{expr}'"
