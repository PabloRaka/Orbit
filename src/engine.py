"""
Project Resonon / PhysLM: Unified Physical Language Model Engine
================================================================
Provides a cohesive, high-level API orchestrating all physical computing
subsystems: Wave Mechanics, Transduction, Associative Memory, Stackless Grammar,
and Equilibrium Propagation Learning.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from src.transducer import GaborWaveTransducer
from src.baseline_phase0 import ContinuousWaveEngine
from src.associative_memory import ContinuousAssociativeMemory
from src.dyck_resonator import PhaseLockingDyckCavity
from src.equilibrium_propagation import MemristiveCrossbarNetwork
from src.bridge import TensorStreamBridge


class PhysLMEngine:
    """
    Unified Physical Language Model Engine.
    """
    def __init__(
        self,
        n_grid: int = 1024,
        x_min: float = -20.0,
        x_max: float = 20.0
    ):
        self.n_grid = n_grid
        self.x_min = x_min
        self.x_max = x_max

        # 1. Transducer (Language <-> Wave Field)
        self.transducer = GaborWaveTransducer(n_grid=n_grid, x_min=x_min, x_max=x_max)

        # 2. Dynamic Wave Propagation Engine (Schrödinger / CGLE)
        self.wave_engine = ContinuousWaveEngine(n_grid=n_grid, x_min=x_min, x_max=x_max)

        # 3. Continuous Associative Memory (Hopfield Infilling)
        self.memory = ContinuousAssociativeMemory(self.transducer)

        # 4. Formal Grammar Cavity Resonator (Stackless Dyck-4)
        self.grammar_cavity = PhaseLockingDyckCavity(max_depth=32)

        # 5. Memristive Learning Crossbar Network (Equilibrium Propagation)
        self.learning_network = MemristiveCrossbarNetwork(
            dim_in=n_grid,
            dim_hid=128,
            dim_out=n_grid,
            eta=0.03,
            beta=0.3,
            dt=0.15
        )

        # 6. Tensor Interoperability Bridge
        self.bridge = TensorStreamBridge()

    def encode(self, text: str) -> np.ndarray:
        """Transforms text into a continuous normalized Hilbert state vector |psi>."""
        return self.transducer.encode(text)

    def decode(self, psi: np.ndarray, expected_length: Optional[int] = None) -> str:
        """Decodes continuous wave state |psi> back to text via quantum projection."""
        return self.transducer.decode(psi, expected_length=expected_length)

    def propagate_wave(self, psi: np.ndarray, steps: int = 50, dt: float = 0.001) -> np.ndarray:
        """Simulates unitary physical wave evolution across time."""
        v_flat = np.zeros(self.n_grid)
        state = psi.copy()
        for _ in range(steps):
            state = self.wave_engine.step_unitary_split_operator(state, v_flat, dt=dt)
        return state

    def register_knowledge(self, concept_text: str) -> None:
        """Registers a knowledge concept into the continuous associative memory."""
        self.memory.store(concept_text)

    def complete(self, prompt: str, target_length: int, steps: int = 120) -> Tuple[str, float]:
        """Performs soft-clamped energy relaxation to complete a partial prompt."""
        return self.memory.complete(prompt, total_expected_length=target_length, steps=steps)

    def check_syntax(self, expr: str) -> Tuple[bool, Dict[str, Any]]:
        """Validates nested grammatical recursion without a digital stack."""
        return self.grammar_cavity.parse(expr)
