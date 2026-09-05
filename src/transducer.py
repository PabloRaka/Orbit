"""
Project Resonon / PhysLM: Tokenless Harmonic Wave Transducer
============================================================
Translates raw text into continuous physical wave packets |psi(x)> and
decodes continuous physical fields back into text via quantum projection.

Physical Foundation (Layer 2 & 3):
    1. Dual-Harmonic Formant Quantization:
       Mimics biological vocal tract acoustics and solid-state phonon modes.
       Each character c in ASCII[32..126] is mapped to a pair of orthogonal
       harmonic frequencies (F1, F2) on a localized window of width W:
           c -> (m, n), where m = idx % 10, n = idx // 10
           k_1(m) = (1 + m) * delta_k
           k_2(n) = (12 + n) * delta_k
           delta_k = 2 * pi / W

    2. Spatial Sequence Layout:
       Characters are placed at continuous spatial coordinates x_j = x_start + j * spacing.
       Localized Gaussian envelope prevents frequency aliasing while allowing natural
       phase continuity between adjacent concepts.

    3. Projective Quantum Measurement (Decoding):
       At each position window x_j, the field is projected onto candidate basis states:
           c_j = argmax_c |<phi_{j, c} | psi>|
"""

import numpy as np
from typing import Optional


class GaborWaveTransducer:
    def __init__(
        self,
        n_grid: int = 1024,
        x_min: float = -20.0,
        x_max: float = 20.0,
        window_width: float = 1.4,
        char_spacing: float = 1.5,
        sigma: float = 0.4
    ):
        self.n_grid = n_grid
        self.x_min = x_min
        self.x_max = x_max
        self.x = np.linspace(x_min, x_max, n_grid, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        
        self.window_width = window_width
        self.char_spacing = char_spacing
        self.sigma = sigma
        
        # Fundamental orthogonal wavenumber increment: delta_k = 2*pi / W
        self.delta_k = 2.0 * np.pi / window_width

        # Printable ASCII character set (95 characters: 32 to 126)
        self.charset = [chr(i) for i in range(32, 127)]
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}

    def _char_formants(self, char: str) -> tuple[float, float]:
        """Returns dual harmonic frequencies (k_low, k_high) for character."""
        idx = self.char_to_idx.get(char, 0)
        m = idx % 10
        n = idx // 10
        k_low = (1 + m) * self.delta_k
        k_high = (12 + n) * self.delta_k
        return k_low, k_high

    def basis_probe(self, x_j: float, char: str) -> np.ndarray:
        """
        Constructs normalized quantum probe state <phi_{j, c}| centered at x_j.
        """
        k_low, k_high = self._char_formants(char)
        dist = self.x - x_j
        envelope = np.exp(-(dist ** 2) / (2.0 * (self.sigma ** 2)))
        carrier = 0.5 * (np.exp(1j * k_low * self.x) + np.exp(1j * k_high * self.x))
        probe = envelope * carrier
        norm = np.sqrt(np.sum(np.abs(probe) ** 2) * self.dx)
        return probe / norm if norm > 1e-12 else probe

    def encode(self, text: str) -> np.ndarray:
        """
        Transforms text string into continuous wave packet psi(x) in Hilbert space.
        """
        if not text:
            return np.zeros(self.n_grid, dtype=complex)

        length = len(text)
        total_span = (length - 1) * self.char_spacing
        x_start = -total_span / 2.0

        psi = np.zeros(self.n_grid, dtype=complex)

        for j, char in enumerate(text):
            x_j = x_start + j * self.char_spacing
            k_low, k_high = self._char_formants(char)
            dist = self.x - x_j
            envelope = np.exp(-(dist ** 2) / (2.0 * (self.sigma ** 2)))
            carrier = 0.5 * (np.exp(1j * k_low * self.x) + np.exp(1j * k_high * self.x))
            psi += envelope * carrier

        # Unitary normalization: int |psi(x)|^2 dx = 1.0
        norm = np.sqrt(np.sum(np.abs(psi) ** 2) * self.dx)
        if norm > 1e-12:
            psi /= norm

        return psi

    def decode(self, psi: np.ndarray, expected_length: Optional[int] = None) -> str:
        """
        Reconstructs text from physical field psi(x) via projective quantum measurements.
        """
        if expected_length is None or expected_length <= 0:
            intensity = np.abs(psi) ** 2
            threshold = 0.05 * np.max(intensity)
            active_x = self.x[intensity > threshold]
            if len(active_x) == 0:
                return ""
            span = np.max(active_x) - np.min(active_x)
            expected_length = max(1, int(round(span / self.char_spacing)) + 1)

        total_span = (expected_length - 1) * self.char_spacing
        x_start = -total_span / 2.0

        decoded_chars = []

        for j in range(expected_length):
            x_j = x_start + j * self.char_spacing
            best_char = "?"
            max_overlap = -1.0

            for char in self.charset:
                probe = self.basis_probe(x_j, char)
                overlap = np.abs(np.sum(np.conj(probe) * psi) * self.dx)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_char = char

            decoded_chars.append(best_char)

        return "".join(decoded_chars)
