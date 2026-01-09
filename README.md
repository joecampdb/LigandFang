# Project LigandFang- LigandSBM: Branched Schrödinger Bridge Matching Demo
<img width="1500" height="1500" alt="image" src="https://github.com/user-attachments/assets/f069ef59-7293-419c-aa3e-9cf1a9eb598a" />

## Overview
This repository contains a proof-of-concept implementation of **Branched Schrödinger Bridge Matching (SBM)** applied to a simplified 2D manifold. The goal of this project is to demonstrate the efficacy of geometric deep learning techniques in modeling multi-modal transport problems with energy constraints, serving as a foundational prototype for ligand conformational sampling.

## Project Scope
This script serves as an algorithmic sandbox to validate specific loss functions and transport dynamics before scaling to high-dimensional biological data. It focuses on:
- **Multi-modal Transport:** Modeling the transition from a single source distribution (unbound state) to multiple distinct target distributions (binding modes).
- **Energy-Constrained Path Planning:** Learning trajectories that minimize kinetic energy while avoiding high-cost regions (simulating steric clashes or energy barriers).
- **Flow Matching:** distilling optimal transport plans into a continuous-time velocity field for efficient sampling.

## Technical Implementation
The implementation is purely geometric and operates on a 2D Euclidean space to allow for rapid iteration and visualization.

### Key Components
1.  **`GeoPathMLP` (Geodesic Interpolant):** A neural network that learns non-linear deviations from linear interpolation. It effectively minimizes an energy functional composed of:
    *   *Kinetic Energy:* Promoting smoothness and efficiency.
    *   *Potential Energy:* An obstacle cost function (Gaussian formulation) representing forbidden regions.

2.  **`VelocityNet` (Flow Matching):** A secondary network trained to regress the vector field of the optimal paths learned by the Geodesic Interpolant. This allows for the generation of new samples via Ordinary Differential Equation (ODE) integration.

3.  **`BranchSBM`:** The orchestrator class that manages the branched transport logic, handling the splitting of probability mass from the source to multiple targets ($x_0 \to x_{1A}, x_{1B}$).

### Synthetic Data
To isolate algorithmic performance from data noise, the system uses synthetic distributions:
- **Source:** A Gaussian cluster representing an initial state (e.g., a generic docking funnel).
- **Targets:** Two distinct Gaussian clusters representing stable endpoints (e.g., specific binding pockets or conformers).
- **Obstacle:** A fixed high-cost region at the origin that trajectories must learn to circumnavigate.

## Usage
The script is self-contained and can be run directly to perform training and visualization.

```bash
python ligandsbm.py
```

### Outputs
Upon execution, the script produces:
- Training logs for both the Geodesic and Flow matching phases.
- `ligandsbm_demo.png`: A visualization of the learned vector fields and particle trajectories maneuvering around the obstacle to reach the target modes.

## Future Directions
This prototype lays the groundwork for:
- Scale-up to $SE(3)$ equivariant networks for rigid body transformations.
- Integration with molecular featurizers (e.g., RDKit, GNNs) to replace synthetic 2D points with high-dimensional conformer embeddings.
- Application to protein-ligand docking scenarios where "obstacles" represent protein backbones.
