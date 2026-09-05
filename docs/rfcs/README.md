# Project Resonon / PhysLM: RFC Index
## Modular Technical Specifications & Subsystems

This directory houses granular **Requests for Comments (RFCs)** that extend the core [Backbone Specifications](../backbone/README.md) into concrete, verified subsystem implementations.

---

## Active RFC Registry

| RFC | Title | Subsystem Area | Implementation Status |
| :--- | :--- | :--- | :--- |
| **[RFC-001](RFC-001_TOKENLESS_SPECTRAL_TRANSDUCER.md)** | **Tokenless Continuous Spectral Transducer** | Input/Output Interface | `ACTIVE` (Verified in `src/transducer.py`) |
| **[RFC-002](RFC-002_SYMPLECTIC_HAMILTONIAN_INTEGRATORS.md)** | **Symplectic Hamiltonian Integrators** | Wave Engine & Solvers | `ACTIVE` (Verified in `src/baseline_phase0.py`) |
| **[RFC-003](RFC-003_EQUILIBRIUM_PROPAGATION_MEMRISTIVE_CROSSBAR.md)** | **Equilibrium Propagation on Crossbars** | Learning / No-Backprop | `ACTIVE` (Verified in `src/equilibrium_propagation.py`) |
| **[RFC-004](RFC-004_THERMAL_NOISE_BOLTZMANN_SAMPLING.md)** | **Thermal Johnson-Nyquist Noise Sampling** | Hardware & Stochastic Engine | `ACTIVE` (Calibrated in `docs/benchmarks/`) |

---

## RFC Lifecycle States
- `DRAFT`: Under active discussion and formulation.
- `ACTIVE`: Approved, implemented, and verified against tests.
- `STABILIZED`: Empirically calibrated against hardware silicon/optics.
- `SUPERSEDED`: Replaced by an updated physical specification.
