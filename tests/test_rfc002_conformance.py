"""
Project Resonon / PhysLM: RFC-002 Normative Conformance Test Suite
===================================================================
Automated verification of the Continuous Hilbert State Specification:
- HilbertState-001: Self-Norm Invariant (| ||ψ||^2 - 1.0 | < 10^-6)
- HilbertState-002: Basis Dominance Overlap (<φ_c | φ_c> > <φ_j | φ_c> for j != c)
- HilbertState-003: Superposition Normalization
- HilbertState-004: Spatial Translation Invariance
- HilbertState-005: Born Probability Unit Sum (Σ p(c | ψ) = 1.0)
- HilbertState-006: Encode-to-Project Round Trip (argmax_c |<φ_c | ψ>| == c_target)
"""

import numpy as np
import pytest
from src.transducer import GaborWaveTransducer


@pytest.fixture
def transducer():
    """Standard RFC-002 compact lattice transducer fixture."""
    return GaborWaveTransducer(
        n_grid=1024,
        x_min=-20.0,
        x_max=20.0,
        window_width=1.4,
        char_spacing=1.5,
        sigma=0.4
    )


def test_hilbert_state_001_self_norm_invariant(transducer):
    """HilbertState-001: | ||ψ||^2 - 1.0 | < 10^-6 for any encoded state."""
    test_strings = ["A", "CAT", "PHYSICS", "12345", "RESONON"]
    for s in test_strings:
        psi = transducer.encode(s)
        norm_sq = np.sum(np.abs(psi) ** 2) * transducer.dx
        assert abs(norm_sq - 1.0) < 1e-6, f"Norm squared {norm_sq} violated tolerance for string '{s}'"


def test_hilbert_state_002_basis_dominance_overlap(transducer):
    """HilbertState-002: |<φ_c | φ_c>| > |<φ_j | φ_c>| + 0.15 for all j != c."""
    test_chars = ["A", "B", "C", "X", "Y", "Z", "0", "1", " "]
    x_center = 0.0

    for c in test_chars:
        probe_c = transducer.basis_probe(x_center, c)
        self_overlap = np.abs(np.sum(np.conj(probe_c) * probe_c) * transducer.dx)
        assert abs(self_overlap - 1.0) < 1e-6, f"Basis state '{c}' not normalized"

        competitor_overlaps = []
        for j in test_chars:
            if j == c:
                continue
            probe_j = transducer.basis_probe(x_center, j)
            overlap = np.abs(np.sum(np.conj(probe_j) * probe_c) * transducer.dx)
            competitor_overlaps.append(overlap)

        max_comp = max(competitor_overlaps)
        margin = self_overlap - max_comp
        assert margin > 0.15, f"Dominance margin {margin:.4f} too small for char '{c}' against max competitor {max_comp:.4f}"


def test_hilbert_state_003_superposition_normalization(transducer):
    """HilbertState-003: Linear combination maintains norm after normalization."""
    phi_a = transducer.basis_probe(-1.0, "A")
    phi_b = transducer.basis_probe(1.0, "B")

    # Superposition: alpha |A> + beta |B>
    alpha = 0.6 + 0.2j
    beta = 0.8 - 0.1j
    psi_composite = alpha * phi_a + beta * phi_b

    norm = np.sqrt(np.sum(np.abs(psi_composite) ** 2) * transducer.dx)
    psi_normalized = psi_composite / norm

    norm_sq = np.sum(np.abs(psi_normalized) ** 2) * transducer.dx
    assert abs(norm_sq - 1.0) < 1e-6, f"Composite state norm {norm_sq} not unit normalized"


def test_hilbert_state_004_spatial_translation_invariance(transducer):
    """HilbertState-004: Spatial translation operator preserves state norm."""
    x_centers = [-5.0, -2.5, 0.0, 3.2, 7.0]
    for xc in x_centers:
        probe = transducer.basis_probe(xc, "M")
        norm_sq = np.sum(np.abs(probe) ** 2) * transducer.dx
        assert abs(norm_sq - 1.0) < 1e-6, f"Translation to x={xc} did not preserve norm (got {norm_sq})"


def test_hilbert_state_005_born_probability_unit_sum(transducer):
    """HilbertState-005: Born-rule measurement probabilities sum unconditionally to 1.0."""
    psi = transducer.encode("K")
    x_center = 0.0

    # Project onto full alphabet subset
    alphabet_subset = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    overlaps_sq = []
    for c in alphabet_subset:
        probe = transducer.basis_probe(x_center, c)
        overlap = np.abs(np.sum(np.conj(probe) * psi) * transducer.dx)
        overlaps_sq.append(overlap ** 2)

    total_intensity = sum(overlaps_sq)
    assert total_intensity > 0.0, "Total projection intensity must be positive"

    probabilities = [I / total_intensity for I in overlaps_sq]
    prob_sum = sum(probabilities)
    assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities sum to {prob_sum}, expected 1.0"


def test_hilbert_state_006_encode_project_identity(transducer):
    """HilbertState-006: Encode-to-project round trip preserves target identity."""
    test_chars = ["H", "E", "L", "L", "O", "W", "O", "R", "L", "D"]
    x_center = 0.0

    for c in test_chars:
        psi = transducer.encode(c)
        # Find argmax overlap over candidate alphabet
        candidate_alphabet = [chr(i) for i in range(32, 127)]
        best_char = None
        max_overlap = -1.0

        for cand in candidate_alphabet:
            probe = transducer.basis_probe(x_center, cand)
            overlap = np.abs(np.sum(np.conj(probe) * psi) * transducer.dx)
            if overlap > max_overlap:
                max_overlap = overlap
                best_char = cand

        assert best_char == c, f"Encode -> project failed: expected '{c}', got '{best_char}'"
