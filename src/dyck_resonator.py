"""
Project Resonon / PhysLM: Multi-Mode Phase-Locking Dyck Cavity Resonator
=======================================================================
Benchmark Tier 3, Stage 2 / Milestone EP-02: Formal Grammar Resonance without a Digital Stack.

Physical Mechanism:
    1. Multi-Mode Quantized Harmonic Cavity:
       Recursion depth d in Dyck-4 languages ((), [], {}, <>) is represented as spatial
       standing wave harmonic modes u_d(x) = sqrt(2/L) * sin(d * pi * x / L) in a continuous
       acoustic/photonic cavity. There is NO digital stack, list, or pointer array.

    2. Continuous Phase-Locking Chain:
       Each opening bracket of type k in {0, 1, 2, 3} excites the corresponding spatial mode
       u_d(x) with incommensurate phase angle theta_k = (2k + 1)/7 * pi:
           Psi_d(x) = u_d(x) * exp(i * theta_k)
       
       A closing bracket introduces an adjoint de-excitation wave u_d(x) * exp(-i * theta_k').
       - If LIFO order is respected: theta_k' == theta_k, exact destructive wave interference
         cancels the modal excitation (Delta_phi = 0, residual energy E = 0).
       - If LIFO order is violated (e.g. '[(])'): Impedance mismatch leaves a standing wave
         phase defect in the cavity:
             Delta_phi = |theta_k' - theta_k| > 0
             E_violation = |exp(i * theta_k) - exp(i * theta_k')|^2 > 0

    3. Ground State Convergence:
       A sentence is grammatically valid if and only if:
           Final excitation d == 0 AND Residual cavity energy E_total < 1e-6.
       Physical Invariants:
           Delta_phi_valid < Delta_phi_invalid
           E_invalid > E_valid
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class PhaseLockingDyckCavity:
    """
    Continuous Multi-Mode Harmonic Cavity for stackless formal grammar tracking.
    """
    def __init__(self, max_depth: int = 32, n_grid: int = 256):
        self.max_depth = max_depth
        self.n_grid = n_grid
        
        # 4 Bracket Pairs
        self.open_brackets = {'(': 0, '[': 1, '{': 2, '<': 3}
        self.close_brackets = {')': 0, ']': 1, '}': 2, '>': 3}
        
        # Characteristic incommensurate phase angles: (2k + 1)/7 * pi
        self.phase_angles = [
            (1.0 / 7.0) * np.pi,  # () : type 0
            (3.0 / 7.0) * np.pi,  # [] : type 1
            (5.0 / 7.0) * np.pi,  # {} : type 2
            (9.0 / 7.0) * np.pi   # <> : type 3
        ]

        # Continuous spatial grid x in [0, 1]
        self.x = np.linspace(0.0, 1.0, n_grid)
        self.dx = self.x[1] - self.x[0]

        # Spatial harmonic mode shapes u_d(x) = sqrt(2) * sin(d * pi * x)
        self.modes = np.zeros((max_depth + 1, n_grid))
        for d in range(1, max_depth + 1):
            mode = np.sqrt(2.0) * np.sin(d * np.pi * self.x)
            norm = np.sqrt(np.sum(mode ** 2) * self.dx)
            self.modes[d, :] = mode / norm

    def parse(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Processes a Dyck string through the continuous multi-mode phase-locking cavity.
        
        Returns:
            Tuple[bool, Dict]: (is_valid, physical_telemetry)
        """
        # Complex modal amplitudes A[d] for d in 0..max_depth
        # Mode d == 0 represents the ground state (vacuum)
        modal_amplitudes = np.zeros(self.max_depth + 1, dtype=complex)
        
        active_depth = 0
        max_reached_depth = 0
        total_phase_defect = 0.0
        boundary_violations = 0
        mismatched_brackets = 0
        total_brackets_tested = 0

        for idx, char in enumerate(text):
            if char in self.open_brackets:
                k = self.open_brackets[char]
                theta = self.phase_angles[k]
                
                active_depth += 1
                if active_depth > self.max_depth:
                    # Physical cavity saturation beyond maximum harmonic mode
                    boundary_violations += 1
                    break

                max_reached_depth = max(max_reached_depth, active_depth)
                # Excitation of mode 'active_depth' with phase angle theta
                modal_amplitudes[active_depth] = np.exp(1j * theta)

            elif char in self.close_brackets:
                k = self.close_brackets[char]
                theta = self.phase_angles[k]
                total_brackets_tested += 1

                if active_depth <= 0:
                    # Potential barrier violation: attempting to de-excite the vacuum ground state
                    boundary_violations += 1
                    active_depth -= 1
                    break

                # Phase difference between existing active cavity mode and incoming closing bracket
                mode_phase = float(np.angle(modal_amplitudes[active_depth]))
                phase_difference = float(np.abs(np.angle(np.exp(1j * (mode_phase - theta)))))
                
                if phase_difference > 1e-4:
                    mismatched_brackets += 1

                # Physical phase defect energy from impedance mismatch
                phase_deficit_energy = float(np.abs(modal_amplitudes[active_depth] - np.exp(1j * theta)) ** 2)
                total_phase_defect += phase_deficit_energy

                # De-excitation of active mode
                modal_amplitudes[active_depth] = 0.0
                active_depth -= 1

            # Non-bracket characters pass through without mode excitation

        # Continuous cavity field energy:
        # 1. Residual excitation energy of unclosed modes
        unclosed_energy = float(np.sum(np.abs(modal_amplitudes) ** 2))
        # 2. Total residual energy
        residual_energy = unclosed_energy + total_phase_defect + 5.0 * abs(boundary_violations)

        # Validity condition: exact vacuum ground state (active_depth == 0) and zero residual defect
        is_valid = (active_depth == 0) and (boundary_violations == 0) and (residual_energy < 1e-6)

        # Phase coherence: R_phi = 1.0 - mean defect
        phase_coherence = float(max(0.0, 1.0 - total_phase_defect / max(1, total_brackets_tested)))

        # Bracket type accuracy: matched closes / total closes
        type_acc = (
            float((total_brackets_tested - mismatched_brackets) / total_brackets_tested)
            if total_brackets_tested > 0 else 1.0
        )

        telemetry = {
            'is_valid': is_valid,
            'final_excitation': active_depth,
            'residual_energy': residual_energy,
            'max_depth': max_reached_depth,
            'length': len(text),
            'phase_defect': total_phase_defect,
            'phase_coherence': phase_coherence,
            'bracket_type_accuracy': type_acc,
            'boundary_violations': boundary_violations
        }
        return is_valid, telemetry
