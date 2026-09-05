# Architectural Backbone (Ground Truth)
## Core Mathematical and Physical Foundations

This directory contains the foundational specifications for the Physical Language Model (PhysLM / Project Resonon). These documents define the core physics, mathematical invariants, software architecture, and hardware implementation path.

---

## Document Index

1. **[01 - Vision & Grand Architecture](01_VISION_AND_GRAND_ARCHITECTURE.md)**
   * Why Transformers are silicon/GPU artifacts.
   * The Triadic Engine: Quantum (Representation), Wave (Evolution), Thermal (Optimization/Sampling).
   * 3-Layer Representation Pipeline & 3-Tiered Learning Hierarchy.

2. **[02 - Mathematical & Physical Formulation](02_MATHEMATICAL_AND_PHYSICAL_FORMULATION.md)**
   * Complex Hilbert Space $\mathcal{H}$ state definitions.
   * Non-linear Complex Ginzburg-Landau / Gross-Pitaevskii field equations.
   * Exact loss functions for phase coherence, wave dynamics, and variational free energy.
   * Local parameter updates via Equilibrium Propagation (bypassing global backprop).

3. **[03 - Software Simulation Specification](03_SOFTWARE_SIMULATION_SPECIFICATION.md)**
   * Computational architecture using **JAX** and **Diffrax**.
   * Differentiable ODE/SDE solvers and complex field operators.
   * Zero-copy PyTorch bridge via DLPack.
   * Minimal self-contained simulation testbench pattern.

4. **[04 - Hardware Roadmap & Physical Mapping](04_HARDWARE_ROADMAP_AND_MAPPING.md)**
   * Physical substrate mapping: Silicon Photonics, Memristor crossbars, and Johnson-Nyquist thermal noise.
   * 4-Phase Roadmap: Software Simulation $\to$ Analog CMOS $\to$ Tri-Substrate Hybrid $\to$ Coherent All-Photonic.
   * Analysis of analog drift, precision limits, and I/O transduction.
