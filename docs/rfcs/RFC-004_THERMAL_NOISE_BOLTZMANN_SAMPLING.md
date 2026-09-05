# RFC-004: Thermal Johnson-Nyquist Noise for Boltzmann Sampling
## Subsystem Specification: Native Physical Sampling & Thermodynamic Creativity

* **Status:** `ACTIVE` (Empirically Calibrated in [`src/baseline_phase0.py`](../../src/baseline_phase0.py))
* **Author:** Project Resonon / PhysLM
* **Base Document:** [Hardware Roadmap 04](../backbone/04_HARDWARE_ROADMAP_AND_MAPPING.md)

---

## 1. Problem Statement

In digital LLMs, non-deterministic sampling requires:
1. Software floating-point evaluation of the exponential softmax distribution $P_i \propto e^{z_i / T}$.
2. Synthetic pseudo-random number generator (PRNG) algorithms that consume digital clock cycles and memory bandwidth.

---

## 2. Specification: Johnson-Nyquist Thermal SDE

RFC-004 formalizes generative sampling as a physical **Langevin stochastic relaxation** driven by real thermal noise:

$$d\psi(\mathbf{x}, \tau) = -\frac{\delta \mathcal{F}}{\delta \psi^*(\mathbf{x})} d\tau + \sqrt{2 k_B T_{\text{phys}}} \, d\mathbf{W}(\mathbf{x}, \tau)$$

### 2.1 Physical Noise Generator (Hardware Mapping)
In an analog/photonic co-processor, the stochastic driving term $d\mathbf{W}$ is harvested directly from unamplified Johnson-Nyquist thermal voltage fluctuations across an integrated resistor:

$$S_v(f) = 4 k_B T R \quad (\text{V}^2/\text{Hz})$$

* Increasing "Temperature" ($T$) physically scales the thermal fluctuation amplitude $\sigma_{\text{thermal}} = \sqrt{2 k_B T R \Delta f}$.
* The system naturally explores the phase space of semantic attractor valleys. Higher physical temperature increases the probability of tunneling over local energy barriers into novel creative conceptual basins, while low temperature locks the trajectory into the ground state.

### 2.2 Topological Robustness Margin
Based on empirical calibration in [Phase 0 Results](../benchmarks/02_PHASE0_EMPIRICAL_BASELINE_RESULTS.md):
* Under continuous thermal noise injection ($\sigma = 0.05$), semantic attractor drift is bounded to $\Delta x < 0.009$.
* The model remains **topologically stable against analog thermal fluctuations**, eliminating the need for complex digital error-correction codes during physical sampling.
