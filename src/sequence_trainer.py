"""
Project Resonon / PhysLM: Autoregressive Sequence Trainer & Generative Sampler
=============================================================================
Translates continuous text sequences into physical wave transitions:
    psi_context(x) -> [Crossbar EqProp] -> psi_target(x) -> [Hilbert-Space Basis Projection] -> c_next

Physical Principles (Layer 3 & 4):
    1. Causal Sequence Windowing:
       Extracts sliding-window text transitions (c_{t-K..t}, c_{t+1}) and transforms
       them into continuous Hilbert state vectors using GaborWaveTransducer.
    2. Local Equilibrium Propagation:
       Optimizes memristive crossbar conductances using contrastive physical relaxation
       (Free Phase -> Nudged Phase) without computational graphs or backpropagation.
    3. Complex Hilbert-Space Projection & Born-Rule-Inspired Sampling (RFC-004):
       Decodes predicted continuous wave fields into discrete characters via Hilbert-space
       basis projection with optional Johnson-Nyquist thermal noise injection:
           P(c) ~ exp( |<phi_c | psi>|^2 / T )
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from src.transducer import GaborWaveTransducer
from src.equilibrium_propagation import MemristiveCrossbarNetwork


class PhysicalCrossbarLayer:
    """
    Direct single-layer memristive crossbar array with local contrastive Equilibrium Propagation updates.
    Operates directly on continuous wave fields W in C^{dim_out x dim_in}.
    """
    def __init__(self, dim_in: int, dim_out: int, eta: float = 0.08, beta: float = 0.35):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.eta = eta
        self.beta = beta
        scale = 0.05 / np.sqrt(dim_in)
        self.W = scale * (np.random.normal(0, 1, (dim_out, dim_in)) + 1j * np.random.normal(0, 1, (dim_out, dim_in)))

    def predict(self, x: np.ndarray, steps: int = 1) -> np.ndarray:
        return self.W @ x

    def train_step(self, x: np.ndarray, target: np.ndarray, **kwargs) -> float:
        # Free phase
        y_0 = self.predict(x)
        loss = float(np.mean(np.abs(y_0 - target) ** 2))

        # Nudged phase
        y_beta = y_0 + self.beta * (target - y_0)

        # Contrastive update
        coeff = self.eta / self.beta
        dW = coeff * np.outer(y_beta - y_0, np.conj(x))
        self.W += dW
        return loss


class CausalSequenceDataset:
    """
    Extracts causal sliding-window character transitions from text corpora
    and prepares continuous Hilbert wave representations.
    """
    def __init__(self, transducer: GaborWaveTransducer, context_window: int = 1):
        self.transducer = transducer
        self.context_window = max(1, context_window)

    def extract_transitions(self, text: str) -> List[Tuple[str, str]]:
        """Extracts (context_str, target_char) pairs from raw text."""
        transitions = []
        if len(text) <= self.context_window:
            return transitions

        for i in range(len(text) - self.context_window):
            context = text[i : i + self.context_window]
            target = text[i + self.context_window]
            transitions.append((context, target))
        return transitions

    def encode_transition(self, context: str, target: str) -> Tuple[np.ndarray, np.ndarray]:
        """Encodes context and target character into continuous wave packets."""
        psi_ctx = self.transducer.encode(context)
        # Target character is encoded centered at x=0
        psi_target = self.transducer.encode(target)
        return psi_ctx, psi_target


class AutoregressiveSequenceTrainer:
    """
    Autoregressive Sequence Trainer and Generator using Equilibrium Propagation
    and continuous quantum projective measurements.
    """
    def __init__(
        self,
        transducer: Optional[GaborWaveTransducer] = None,
        network: Optional[Any] = None,
        context_window: int = 1,
        dim_hid: int = 128,
        layer_type: str = "recurrent"
    ):
        self.transducer = transducer or GaborWaveTransducer(n_grid=256, x_min=-10.0, x_max=10.0)
        self.context_window = context_window
        self.dataset = CausalSequenceDataset(self.transducer, context_window=context_window)

        dim_grid = self.transducer.n_grid
        if network is not None:
            self.network = network
        elif layer_type == "direct":
            self.network = PhysicalCrossbarLayer(dim_in=dim_grid, dim_out=dim_grid, eta=0.08, beta=0.35)
        else:
            self.network = MemristiveCrossbarNetwork(
                dim_in=dim_grid,
                dim_hid=dim_hid,
                dim_out=dim_grid,
                eta=0.03,
                beta=0.3,
                dt=0.15
            )

        # Precompute vectorized quantum basis probes for single-character decoding at x=0
        self._precompute_basis_probes()


    def _precompute_basis_probes(self) -> None:
        """
        Precomputes matrix of candidate basis probes at x=0 across all printable ASCII chars.
        P[c, :] = conj(probe_c) * dx
        Overlap vector for any wave psi: overlaps = abs(P @ psi)
        Accelerates single-character quantum measurement by >100x compared to iterative probe construction.
        """
        n_chars = len(self.transducer.charset)
        n_grid = self.transducer.n_grid
        self.probe_matrix = np.zeros((n_chars, n_grid), dtype=complex)

        for idx, char in enumerate(self.transducer.charset):
            probe = self.transducer.basis_probe(x_j=0.0, char=char)
            self.probe_matrix[idx, :] = np.conj(probe) * self.transducer.dx

    def measure_wave(self, psi: np.ndarray, temperature: float = 0.0) -> Tuple[str, np.ndarray]:
        """
        Projects wave state onto character basis states in Hilbert space.
        If temperature == 0.0: deterministic projective measurement (argmax).
        If temperature > 0.0: Born-rule-inspired Boltzmann sampling (RFC-004).
        """
        # Vectorized Hilbert-space overlap: O_c = |<phi_c | psi>|
        overlaps = np.abs(self.probe_matrix @ psi)

        if temperature <= 1e-6:
            best_idx = int(np.argmax(overlaps))
            return self.transducer.charset[best_idx], overlaps

        # Boltzmann thermodynamic distribution: P(c) ~ exp(O_c / T)
        # Scaled to avoid numerical overflow
        scaled = (overlaps - np.max(overlaps)) / max(temperature, 1e-4)
        probs = np.exp(scaled)
        prob_sum = np.sum(probs)
        if prob_sum > 1e-12:
            probs /= prob_sum
        else:
            probs = np.ones(len(probs)) / len(probs)

        sampled_idx = int(np.random.choice(len(self.transducer.charset), p=probs))
        return self.transducer.charset[sampled_idx], overlaps

    def train_epoch(
        self,
        transitions: List[Tuple[str, str]],
        free_steps: int = 25,
        nudge_steps: int = 12,
        batch_mode: bool = False
    ) -> float:
        """
        Performs one epoch of local Equilibrium Propagation updates across all transitions.
        Supports both online physical crossbar updates (default) and batch-accumulated updates.
        """
        if not transitions:
            return 0.0

        if not batch_mode:
            losses = []
            for context, target in transitions:
                psi_ctx, psi_target = self.dataset.encode_transition(context, target)
                loss = self.network.train_step(
                    psi_ctx,
                    psi_target,
                    free_steps=free_steps,
                    nudge_steps=nudge_steps
                )
                losses.append(loss)
            return float(np.mean(losses))

        # Batch-accumulated physical conductance updates (prevents catastrophic forgetting)
        accum_dW_out = np.zeros_like(self.network.W_out)
        accum_dW_in = np.zeros_like(self.network.W_in)
        accum_dW_rec = np.zeros_like(self.network.W_rec)
        losses = []

        coeff = self.network.eta / self.network.beta

        for context, target in transitions:
            psi_ctx, psi_target = self.dataset.encode_transition(context, target)
            h_0, y_0 = self.network.relax(psi_ctx, steps=free_steps)
            loss = float(np.mean(np.abs(y_0 - psi_target) ** 2))
            losses.append(loss)

            h_beta, y_beta = self.network.relax(
                psi_ctx,
                target=psi_target,
                h_init=h_0,
                y_init=y_0,
                steps=nudge_steps
            )

            accum_dW_out += (np.outer(y_beta, np.conj(h_beta)) - np.outer(y_0, np.conj(h_0)))
            accum_dW_in += (np.outer(h_beta, np.conj(psi_ctx)) - np.outer(h_0, np.conj(psi_ctx)))
            accum_dW_rec += (np.outer(h_beta, np.conj(h_beta)) - np.outer(h_0, np.conj(h_0)))

        M = len(transitions)
        scale = coeff / M
        self.network.W_out += scale * accum_dW_out
        self.network.W_in += scale * accum_dW_in
        dW_rec_sym = 0.5 * (accum_dW_rec + np.conj(accum_dW_rec).T)
        self.network.W_rec += scale * dW_rec_sym

        # Conductance decay / physical leakage regularization
        self.network.W_in *= 0.999
        self.network.W_rec *= 0.999
        self.network.W_out *= 0.999

        # Contractive Lipschitz bound on recurrence to prevent reservoir runaway
        rec_norm = np.linalg.norm(self.network.W_rec, 2)
        if rec_norm > 0.85:
            self.network.W_rec *= (0.85 / rec_norm)

        return float(np.mean(losses))

    def evaluate_accuracy(self, transitions: List[Tuple[str, str]]) -> float:
        """Evaluates top-1 character prediction accuracy across transitions."""
        if not transitions:
            return 0.0
        correct = 0
        for context, target in transitions:
            predicted_char, _ = self.predict_next(context, temperature=0.0)
            if predicted_char == target:
                correct += 1
        return correct / len(transitions)

    def train(
        self,
        corpus: str,
        epochs: int = 20,
        free_steps: int = 25,
        nudge_steps: int = 12,
        batch_mode: bool = False,
        log_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Trains the memristive crossbar network on text corpus via Equilibrium Propagation.
        """
        transitions = self.dataset.extract_transitions(corpus)
        if not transitions:
            raise ValueError(f"Corpus length must be greater than context window ({self.context_window})")

        start_time = time.perf_counter()
        history: List[Dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            epoch_loss = self.train_epoch(
                transitions,
                free_steps=free_steps,
                nudge_steps=nudge_steps,
                batch_mode=batch_mode
            )
            accuracy = self.evaluate_accuracy(transitions)

            entry = {
                "epoch": epoch,
                "loss": epoch_loss,
                "accuracy": accuracy
            }
            history.append(entry)

            if log_callback:
                log_callback(entry)

        total_time = time.perf_counter() - start_time
        return {
            "total_epochs": epochs,
            "transitions_count": len(transitions),
            "total_time_sec": total_time,
            "time_per_epoch_ms": (total_time / epochs) * 1000.0,
            "initial_loss": history[0]["loss"],
            "final_loss": history[-1]["loss"],
            "initial_accuracy": history[0]["accuracy"],
            "final_accuracy": history[-1]["accuracy"],
            "history": history
        }

    def predict_next(
        self,
        context: str,
        temperature: float = 0.0,
        steps: int = 25
    ) -> Tuple[str, np.ndarray]:
        """
        Predicts next character given context string.
        Returns predicted character and predicted wave state.
        """
        # Truncate context to context_window
        effective_context = context[-self.context_window:]
        psi_ctx = self.transducer.encode(effective_context)

        # Inference via free physical relaxation on crossbar
        pred_wave = self.network.predict(psi_ctx, steps=steps)

        # Measure wave to collapse onto discrete character
        char, _ = self.measure_wave(pred_wave, temperature=temperature)
        return char, pred_wave

    def generate(
        self,
        seed: str,
        max_chars: int = 30,
        temperature: float = 0.0,
        stop_chars: Optional[List[str]] = None
    ) -> str:
        """
        Autoregressively generates text token-by-token (character-by-character)
        using physical wave propagation and continuous Hilbert-space basis projection feedback.
        """
        generated = seed
        stops = stop_chars or []

        for _ in range(max_chars):
            ctx = generated[-self.context_window:]
            next_char, _ = self.predict_next(ctx, temperature=temperature)
            if next_char in stops:
                break
            generated += next_char

        return generated

    def evaluate_transitions(
        self,
        transitions: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluates single-step next-wave prediction across transitions:
        MSE, top-1 accuracy, and semantic separation margin M = S_target - max_{j != target} S_j.
        """
        if not transitions:
            return {
                "accuracy": 0.0,
                "mean_mse": 0.0,
                "mean_margin": 0.0,
                "mean_s_tgt": 0.0,
                "mean_s_comp": 0.0,
                "count": 0
            }

        correct = 0
        mses = []
        margins = []
        s_tgts = []
        s_comps = []

        for context, target in transitions:
            psi_ctx = self.transducer.encode(context)
            psi_target = self.transducer.encode(target)

            pred_wave = self.network.predict(psi_ctx)
            pred_norm = np.sqrt(np.sum(np.abs(pred_wave) ** 2) * self.transducer.dx)
            if pred_norm > 1e-12:
                pred_wave /= pred_norm

            # MSE in physical Hilbert space
            mse = float(np.sum(np.abs(pred_wave - psi_target) ** 2) * self.transducer.dx)
            mses.append(mse)

            # Vectorized basis overlap
            overlaps = np.abs(self.probe_matrix @ pred_wave)
            target_idx = self.transducer.charset.index(target) if target in self.transducer.charset else -1

            if target_idx >= 0:
                s_tgt = float(overlaps[target_idx])
                comp_overlaps = np.delete(overlaps, target_idx)
                s_comp = float(np.max(comp_overlaps))
                margin = s_tgt - s_comp
                s_tgts.append(s_tgt)
                s_comps.append(s_comp)
                margins.append(margin)

            predicted_char = self.transducer.charset[int(np.argmax(overlaps))]
            if predicted_char == target:
                correct += 1

        return {
            "accuracy": correct / len(transitions),
            "mean_mse": float(np.mean(mses)),
            "mean_margin": float(np.mean(margins)) if margins else 0.0,
            "mean_s_tgt": float(np.mean(s_tgts)) if s_tgts else 0.0,
            "mean_s_comp": float(np.mean(s_comps)) if s_comps else 0.0,
            "count": len(transitions)
        }

    def rollout_with_metrics(
        self,
        seed: str,
        horizon: int,
        mode: str = "projective",  # "projective" (Mode B) or "free_flight" (Mode A)
        reference_seq: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes multi-step autoregressive rollout with detailed physical telemetry:
        - Mode B (projective): measurement intervention resets phase drift at each step.
        - Mode A (free_flight): uncollapsed continuous wave propagation without measurement collapse.
        """
        k = self.context_window
        current_text = seed
        generated_chars = []
        sq_errors = []
        r_phis = []
        d_drifts = []
        d_basises = []
        vcr_matches = 0

        # Slot positions for spatial assembly in free_flight
        total_span = (k - 1) * self.transducer.char_spacing
        x_start = -total_span / 2.0
        x_positions = [x_start + j * self.transducer.char_spacing for j in range(k)]

        # Initial wave slots for free-flight mode
        effective_seed = seed[-k:]
        wave_slots = [self.transducer.encode(c) for c in effective_seed]

        for step in range(horizon):
            if mode == "projective":
                ctx = current_text[-k:]
                psi_ctx = self.transducer.encode(ctx)
                pred_wave = self.network.predict(psi_ctx)
                pred_norm = np.sqrt(np.sum(np.abs(pred_wave) ** 2) * self.transducer.dx)
                if pred_norm > 1e-12:
                    pred_wave /= pred_norm
                overlaps = np.abs(self.probe_matrix @ pred_wave)
                char = self.transducer.charset[int(np.argmax(overlaps))]
                current_text += char
                generated_chars.append(char)
            else:
                # Continuous Analog Free-Flight
                ctx_wave = np.zeros(self.transducer.n_grid, dtype=complex)
                for j in range(k):
                    env = np.exp(-((self.transducer.x - x_positions[j]) ** 2) / (2.0 * (self.transducer.sigma ** 2)))
                    ctx_wave += env * wave_slots[j]
                ctx_norm = np.sqrt(np.sum(np.abs(ctx_wave) ** 2) * self.transducer.dx)
                if ctx_norm > 1e-12:
                    ctx_wave /= ctx_norm

                pred_wave = self.network.predict(ctx_wave)
                pred_norm = np.sqrt(np.sum(np.abs(pred_wave) ** 2) * self.transducer.dx)
                if pred_norm > 1e-12:
                    pred_wave /= pred_norm

                overlaps = np.abs(self.probe_matrix @ pred_wave)
                char = self.transducer.charset[int(np.argmax(overlaps))]
                current_text += char
                generated_chars.append(char)
                wave_slots = wave_slots[1:] + [pred_wave]

            # Observational telemetry
            max_ov = float(np.max(overlaps))
            d_basis = 1.0 - max_ov
            d_basises.append(d_basis)

            # Valid character rate (printable and clear basis alignment)
            if char in self.transducer.charset and max_ov > 0.35:
                vcr_matches += 1

            # Reference comparison (if ground truth reference provided)
            if reference_seq and len(reference_seq) > len(seed) + step:
                ref_char = reference_seq[len(seed) + step]
                ref_wave = self.transducer.encode(ref_char)
                err = float(np.sum(np.abs(pred_wave - ref_wave) ** 2) * self.transducer.dx)
                r_phi = float(np.abs(np.sum(np.conj(ref_wave) * pred_wave) * self.transducer.dx))
                d_drift = 1.0 - r_phi
                sq_errors.append(err)
                r_phis.append(r_phi)
                d_drifts.append(d_drift)

        l_h = float(np.mean(sq_errors)) if sq_errors else 0.0
        e_h = sq_errors[-1] if sq_errors else 0.0
        vcr = (vcr_matches / horizon) * 100.0 if horizon > 0 else 0.0

        return {
            "mode": mode,
            "horizon": horizon,
            "generated_text": current_text,
            "generated_chars": "".join(generated_chars),
            "L_H": l_h,
            "E_H": e_h,
            "VCR": vcr,
            "mean_R_phi": float(np.mean(r_phis)) if r_phis else 0.0,
            "mean_delta_drift": float(np.mean(d_drifts)) if d_drifts else 0.0,
            "mean_delta_basis": float(np.mean(d_basises)) if d_basises else 0.0,
            "trajectory_R_phi": r_phis,
            "trajectory_delta_drift": d_drifts,
            "trajectory_delta_basis": d_basises,
            "trajectory_sq_errors": sq_errors
        }

