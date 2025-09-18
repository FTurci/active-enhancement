# Case Study 4: Molecular Dynamics Simulation Summary
[Code](https://github.com/FTurci/active-enhancement/blob/09795e2bb317a7e7a6b019a038b943cf8fd5ba6e/docs/summer/case_study_4/CS4.ipynb)
## System Setup
- **N = 108** Lennard-Jones particles in simple cubic lattice
- **Density ρ = 0.8442** (reduced units)
- **Target energy per particle = -2.1626** (liquid state)
- **Box volume** calculated from N/ρ, periodic boundaries
- **LJ potential** with cutoff r_cut = 2.5σ and shift correction

## Simulation Details
- **NVE ensemble** (constant number, volume, energy)
- **HOOMD-Blue** framework for MD simulation
- **Timestep dt = 0.002** (reduced units)
- **Neighbour list** with buffer distance
- **Velocity initialisation** to achieve exact target total energy
- **Run length**: 600,000 timesteps

## Key Analyses Performed

### Energy Conservation
- Real-time monitoring of kinetic, potential, and total energy
- Verification of energy conservation in NVE ensemble
- Energy plotting vs time to check stability

### Statistical Error Analysis
- **Flyvbjerg-Petersen method** for block averaging
- Systematic doubling of block sizes to find correlation time
- Error estimation for both kinetic and potential energy
- Temperature calculation from kinetic energy: T = (2/3)⟨KE⟩

### Structural Analysis
- **Time-averaged radial distribution function g(r)**
- Multiple snapshots (100 frames) for statistical averaging
- Comparison with ideal gas reference (g(r) = 1)
- Periodic boundary corrections for distance calculations

## Physical Insights
- **Equilibration** period identification (first 1000 steps excluded)
- **Temperature-energy relationship** in 3D LJ system
- **Liquid structure** characterised by g(r) peaks
- **Energy fluctuations** and their statistical significance

## Computational Methods
- **Exact kinetic energy assignment** via velocity scaling
- **Centre of mass momentum** removal for proper thermostat-free dynamics
- **Manual potential energy calculation** for verification
- **Block averaging algorithm** for correlated data analysis

## Context in Molecular Simulation
This code demonstrates fundamental MD concepts from Frenkel & Smit:
- Setting up realistic liquid-state conditions
- Running microcanonical (NVE) dynamics
- Proper statistical analysis of simulation data
- Understanding energy conservation and fluctuations
- Characterising liquid structure through pair correl