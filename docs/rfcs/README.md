# Project Resonon / PhysLM: RFC Index
## Modular Technical Specifications & Subsystems

This directory houses granular **Requests for Comments (RFCs)** that extend the core [Backbone Specifications](../backbone/README.md) into concrete, verified subsystem implementations.

---

## Phase II Architecture Consolidation Registry

Following the [Experimental Baseline Freeze](../benchmarks/10_EXPERIMENTAL_BASELINE_FREEZE.md), the physical architecture is codified into five formal contracts:

| RFC | Title | Subsystem Area | Status |
| :--- | :--- | :--- | :--- |
| **[RFC-001](RFC-001_PHYSLM_CORE_ARCHITECTURE.md)** | **PhysLM Core Architecture & System Paradigm** | Overall Paradigm & Boundaries | `RATIFIED CONTRACT` |
| **[RFC-002](RFC-002_CONTINUOUS_HILBERT_STATE.md)** | **Continuous Hilbert State Specification** | State, Basis, Metric, Measurement | `RATIFIED CONTRACT` (Verified in `tests/test_rfc002_conformance.py`) |
| **[RFC-003](RFC-003_WAVE_DYNAMICS_ENGINE.md)** | **Wave Dynamics Engine Specification** | Hamiltonian Flow, Cavity, Solvers | `RATIFIED CONTRACT` (Verified in `tests/test_rfc003_conformance.py`) |
| **[RFC-004](RFC-004_THERMAL_BOLTZMANN_ENGINE.md)** | **Thermal / Boltzmann Engine Specification** | Langevin SDE, Thermal Sampling | `RATIFIED CONTRACT` (Verified in `tests/test_rfc004_conformance.py`) |
| **[RFC-005](RFC-005_PHYSICAL_PROTOTYPE_SPECIFICATION.md)** | **Physical Prototype Specification** | Hardware Mapping (Photonics/Crossbar) | `RATIFIED CONTRACT` (Verified in `tests/test_rfc005_conformance.py`) |

---

## Phase I Supporting Subsystem Specifications (Historical)

| RFC | Title | Subsystem Area | Implementation Reference |
| :--- | :--- | :--- | :--- |
| **[RFC-001-Draft](RFC-001_TOKENLESS_SPECTRAL_TRANSDUCER.md)** | **Tokenless Continuous Spectral Transducer** | Input/Output Interface | Verified in `src/transducer.py` |
| **[RFC-002-Draft](RFC-002_SYMPLECTIC_HAMILTONIAN_INTEGRATORS.md)** | **Symplectic Hamiltonian Integrators** | Wave Engine & Solvers | Verified in `src/baseline_phase0.py` |
| **[RFC-003-Draft](RFC-003_EQUILIBRIUM_PROPAGATION_MEMRISTIVE_CROSSBAR.md)** | **Equilibrium Propagation on Crossbars** | Learning / No-Backprop | Verified in `src/equilibrium_propagation.py` |
| **[RFC-004-Draft](RFC-004_THERMAL_NOISE_BOLTZMANN_SAMPLING.md)** | **Thermal Johnson-Nyquist Noise Sampling** | Hardware & Stochastic Engine | Verified in `benchmarks/` |

---

## RFC Lifecycle States
- `DRAFT`: Under active discussion and formulation.
- `PLANNED`: Sequenced on the architectural consolidation roadmap.
- `ACTIVE`: Approved, implemented, and verified against tests.
- `RATIFIED CONTRACT`: Formally ratified architectural invariant derived from empirical evidence.
- `SUPERSEDED`: Replaced by an updated physical specification.
