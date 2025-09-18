# Active Brownian Particle MSD Analysis
[Code](https://github.com/FTurci/active-enhancement/blob/09795e2bb317a7e7a6b019a038b943cf8fd5ba6e/docs/activity_msd/activity_msd.ipynb)

## System Overview

- **Single active particle** (N = 1) in large simulation box (L = 1000)
- **Self-propelled motion** with constant active force magnitude
- **Rotational diffusion** causes random reorientation of propulsion direction
- **HOOMD-Blue** framework with Langevin dynamicsC:\Users\sammu\OneDrive\Documents\Active_Enhancement\Learning hoomd\MSD\MSD.md
## Active Matter Physics

- **Active force**: Constant magnitude along particle's orientation axis
- **Rotational diffusion coefficient**: Controls persistence time of directed motion
- **Translational drag**: Minimal to preserve active dynamics
- **Temperature**: Low (kT = 1) to minimise thermal effects

## Simulation Parameters

- **Multiple independent runs** (typically 200) for statistical averaging
- **High-frequency logging** (every 100 timesteps) for accurate MSD
- **Variable force magnitude** and rotational diffusion rates
- **Long simulation times** (200,000 steps) to observe crossover behaviour

## MSD Analysis Method

- **Ensemble averaging** over multiple independent simulations
- **MSD = ⟨|r(t) - r(0)|²⟩** calculated from particle trajectories
- **Log-log plotting** reveals different motion regimes
- **Statistical robustness** from large number of realisations

## Expected MSD Behaviour

The provided MSD plot shows characteristic active Brownian motion:

- **Short times (t < τ)**: **Ballistic regime** with MSD ∝ t²
    
    - Particle moves ballistically along propulsion direction
    - Slope ≈ 2 on log-log plot
- **Long times (t >> τ)**: **Diffusive regime** with MSD ∝ t
    
    - Rotational diffusion randomises direction
    - Slope ≈ 1 on log-log plot (normal diffusion)
- **Crossover time τ ≈ 1/Dr**: Persistence time of directed motion
    
    - Depends on rotational diffusion coefficient
![[Pasted image 20250917162212.png]]
## 3D Trajectory Visualisation

- **Colour-coded trajectories** showing time progression
- **Start/end markers** to visualise net displacement
- **Persistent motion** segments followed by direction changes
- **Visual confirmation** of active dynamics

## Physical Significance

- Models **self-propelled particles** (bacteria, active colloids, etc.)
- Demonstrates **enhanced diffusion** compared to passive Brownian motion
- Shows **ballistic-to-diffusive crossover** fundamental to active matter
- **Effective diffusion coefficient** depends on activity parameters: $D_{eff} ∝ v^2τ$


## Key Parameters

- **Force magnitude**: Controls propulsion speed
- **Rotational diffusion**: Controls persistence length
- **Ratio v²/Dr**: Determines enhancement over thermal diffusion

