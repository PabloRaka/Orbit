"""
Project Resonon / PhysLM: Continuous Associative Memory & Completion Engine
============================================================================
Implements a continuous Hopfield physical energy landscape for auto-associative
infilling, pattern retrieval, and language completion.

Formulation:
    1. Continuous Stored Memory Field:
       Stores M continuous state vectors {|xi_mu>}_{mu=1}^M in Hilbert space
       encoded via GaborWaveTransducer.

    2. Continuous Hopfield Free Energy:
       E[psi] = - (1/beta) * ln( sum_{mu=1}^M exp( beta * Re<xi_mu | psi> ) ) + (1/2) * ||psi||^2

    3. Dissipative Gradient Flow with Soft Context Clamping:
       d(psi)/d(tau) = - delta E / delta psi* + alpha * P_prompt(psi_prompt - psi) + xi(t)
       where:
           delta E / delta psi* = psi - sum_mu w_mu * xi_mu
           w_mu = exp(beta * Re<xi_mu | psi>) / sum_nu exp(beta * Re<xi_nu | psi>)
           P_prompt is a spatial mask covering the prompt domain.
           alpha is the soft clamping coupling parameter.
"""

import numpy as np
from typing import List, Tuple, Optional
from src.transducer import GaborWaveTransducer


class ContinuousAssociativeMemory:
    def __init__(
        self,
        transducer: GaborWaveTransducer,
        beta: float = 12.0,
        alpha_clamp: float = 0.5,
        dt: float = 0.05
    ):
        """
        Parameters:
            transducer: GaborWaveTransducer instance.
            beta: Inverse temperature parameter for sharp attractor basin separation.
            alpha_clamp: Soft driving coupling strength to anchor prompt context.
            dt: Gradient flow relaxation time step.
        """
        self.transducer = transducer
        self.beta = beta
        self.alpha_clamp = alpha_clamp
        self.dt = dt
        
        self.patterns: List[np.ndarray] = []
        self.pattern_texts: List[str] = []

    def store(self, text: str) -> None:
        """Encodes and registers a concept pattern into the associative physical field."""
        psi_pattern = self.transducer.encode(text)
        self.patterns.append(psi_pattern)
        self.pattern_texts.append(text)

    def compute_overlap(self, psi_a: np.ndarray, psi_b: np.ndarray) -> float:
        """Computes real inner product overlap: Re<psi_a | psi_b>."""
        return float(np.real(np.sum(np.conj(psi_a) * psi_b) * self.transducer.dx))

    def energy(self, psi: np.ndarray) -> float:
        """Calculates total continuous Hopfield energy of current field state."""
        if not self.patterns:
            return 0.0
        overlaps = [self.compute_overlap(p, psi) for p in self.patterns]
        max_ov = max(overlaps)
        # Log-sum-exp stabilization
        log_sum = max_ov + (1.0 / self.beta) * np.log(
            np.sum([np.exp(self.beta * (ov - max_ov)) for ov in overlaps])
        )
        norm_sq = float(np.sum(np.abs(psi) ** 2) * self.transducer.dx)
        return -log_sum + 0.5 * norm_sq

    def complete(
        self,
        prompt_text: str,
        total_expected_length: int,
        steps: int = 150,
        noise_sigma: float = 0.0
    ) -> Tuple[str, float]:
        """
        Performs soft-clamped dissipative relaxation to complete the unmasked sequence.

        Parameters:
            prompt_text: Incomplete prompt string (e.g. 'kucing:').
            total_expected_length: Length of full target sequence (e.g. len('kucing:meong')).
            steps: Number of relaxation steps.
            noise_sigma: Physical thermal noise amplitude.

        Returns:
            Tuple[str, float]: (Decoded completion text, Final energy)
        """
        if not self.patterns:
            return prompt_text, 0.0

        # Construct spatial prompt mask P_prompt(x)
        # Sequence is centered around 0 with total span = (total_expected_length - 1) * spacing
        total_span = (total_expected_length - 1) * self.transducer.char_spacing
        x_start = -total_span / 2.0
        
        prompt_len = len(prompt_text)
        # Boundary coordinate separating prompt region from completion region
        x_prompt_end = x_start + (prompt_len - 0.5) * self.transducer.char_spacing
        
        # Smooth spatial mask: 1.0 on prompt region, decaying to 0.0 on completion region
        prompt_mask = 0.5 * (1.0 - np.tanh((self.transducer.x - x_prompt_end) / self.transducer.sigma))

        # Encode partial prompt text
        # Place characters specifically on the prompt grid coordinates
        psi_prompt = np.zeros(self.transducer.n_grid, dtype=complex)
        for j, char in enumerate(prompt_text):
            x_j = x_start + j * self.transducer.char_spacing
            k_low, k_high = self.transducer._char_formants(char)
            dist = self.transducer.x - x_j
            envelope = np.exp(-(dist ** 2) / (2.0 * (self.transducer.sigma ** 2)))
            carrier = 0.5 * (np.exp(1j * k_low * self.transducer.x) + np.exp(1j * k_high * self.transducer.x))
            psi_prompt += envelope * carrier

        p_norm = np.sqrt(np.sum(np.abs(psi_prompt) ** 2) * self.transducer.dx)
        if p_norm > 1e-12:
            psi_prompt /= p_norm

        # Initialize field state with prompt + low thermal seed
        psi = psi_prompt.copy()
        if noise_sigma > 0.0:
            seed_noise = (np.random.normal(0, noise_sigma, self.transducer.n_grid) + 
                          1j * np.random.normal(0, noise_sigma, self.transducer.n_grid))
            psi += (1.0 - prompt_mask) * seed_noise
            psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * self.transducer.dx)

        # Gradient flow relaxation
        for _ in range(steps):
            # Compute overlaps with all stored patterns
            overlaps = np.array([self.compute_overlap(p, psi) for p in self.patterns])
            max_ov = np.max(overlaps)
            exp_ov = np.exp(self.beta * (overlaps - max_ov))
            weights = exp_ov / np.sum(exp_ov)

            # Continuous Hopfield gradient: delta E / delta psi* = psi - sum_mu w_mu * xi_mu
            attractor_pull = np.zeros(self.transducer.n_grid, dtype=complex)
            for w, pat in zip(weights, self.patterns):
                attractor_pull += w * pat

            # Gradient step towards attractor minimum
            dpsi = (attractor_pull - psi) * self.dt

            # Soft Context Clamping: drive prompt region back towards prompt wave
            dpsi += self.alpha_clamp * prompt_mask * (psi_prompt - psi) * self.dt

            # Thermal Langevin noise
            if noise_sigma > 0.0:
                noise = np.random.normal(0, noise_sigma, self.transducer.n_grid) * np.sqrt(self.dt)
                dpsi += noise

            psi += dpsi

            # Unitary normalization of probability density
            norm = np.sqrt(np.sum(np.abs(psi) ** 2) * self.transducer.dx)
            if norm > 1e-12:
                psi /= norm

        decoded_text = self.transducer.decode(psi, expected_length=total_expected_length)
        final_energy = self.energy(psi)
        return decoded_text, final_energy
