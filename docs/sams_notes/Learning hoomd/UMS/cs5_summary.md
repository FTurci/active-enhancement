# Case Study 5: Transport Properties Summary
[Code](https://github.com/FTurci/active-enhancement/blob/09795e2bb317a7e7a6b019a038b943cf8fd5ba6e/docs/summer/case_study_5/CS5.ipynb)
## System Setup (Identical to CS4)
- **N = 108** LJ particles, **ρ = 0.8442**, target energy per particle = -2.1626
- **NVE ensemble** with exact velocity initialisation
- **HOOMD-Blue** framework, timestep dt = 0.002

## Key Difference: High-Frequency Trajectory Logging
- **TrajectoryLogger** saves positions and velocities every **10 timesteps**
- **50,000 step** equilibration (data discarded)
- **500,000 step** production run for transport property analysis
- Fine time resolution crucial for accurate correlation functions

## Transport Property Calculations

### Mean-Squared Displacement (MSD)
- **MSD = ⟨|r(t) - r(0)|²⟩** with periodic boundary corrections
- Multiple time origins for better statistics
- **Linear regime fitting**: MSD = 6Dt (3D diffusion)
- **Diffusion coefficient**: D = slope/6

### Velocity Auto-Correlation Function (VACF)
- **VACF = ⟨v(0)·v(t)⟩** normalised by ⟨v(0)·v(0)⟩ 
- High-frequency sampling essential for accurate decay
- Exponential decay characterises velocity memory loss

### Green-Kubo Integration
- **Einstein relation**: D = (1/3)∫₀^∞ VACF(t)dt
- Careful **plateau detection** for integration cutoff
- **Trapezoid integration** with running convergence check

## Statistical Methods
- **Multiple time origins** for both MSD and VACF
- **Minimum image convention** for periodic boundaries
- **Variance analysis** to find stable integration plateau
- **Convergence monitoring** of Green-Kubo integral

## Method Comparison
- **Two independent approaches** should agree within ~10-20%
- **MSD method**: Direct from particle trajectories
- **Green-Kubo method**: From microscopic velocity correlations
- **Cross-validation** ensures simulation quality

## Physical Insights
- **Diffusion mechanisms** in liquid-state LJ system
- **Velocity correlation decay** reflects collision dynamics
- **Einstein-Green-Kubo equivalence** demonstration
- **Transport coefficient accuracy** from MD simulations

## Context in Molecular Simulation
This extends CS4 to demonstrate:
- Proper trajectory sampling for transport properties
- Two fundamental approaches to diffusion calculation
- Statistical mechanics relations (fluctuation-dissipation)
- Validation of simulation methodology through method comparison