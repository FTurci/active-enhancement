[Code](https://github.com/FTurci/active-enhancement/blob/037829e5d361d86d9eae1f91a76a44991187e76c/docs/forces/forces.ipynb)
## Custom Force Implementation

- **CentralWell class** inherits from `hoomd.md.force.Custom`
- **Harmonic potential well** centered at simulation box origin
- **Conditional activation** only beyond radius r₀
- **Force calculation**: $F=-k(r-r_0)$ for $r > r_0$, $F = 0$ for $r ≤ r_0$

## Technical Implementation

- **Access simulation state** via `self._state.cpu_local_snapshot`
- **Minimum image convention** for periodic boundary corrections
- **Force arrays modification** through `self.cpu_local_force_arrays`
- **Potential energy contribution** added to energy calculations

## System Setup

- **Simple cubic lattice** of LJ particles in 20×20×20 box
- **Combined forces**: Standard LJ interactions + custom central well
- **NVT ensemble** with Langevin thermostat (kT = 1.0)
- **Parameters**: k = 2.0 (spring constant), r₀ = 3.0 (well radius)

## Simulation Protocol

- **Equilibration**: 5,000 steps
- **Production**: 50,000 steps
- **GSD trajectory output** every 100 steps
- **Fresnel visualization** for particle positions

## Analysis: Radial Density Profile

- **Spherical shell analysis** from simulation center
- **Density calculation**: particles per unit volume vs. distance
- **Equilibration frames skipped** for accurate statistics
- **Expected result**: Higher density near center due to confining potential

## Learning Objectives

- **Custom force development** in HOOMD-Blue framework
- **Force-energy consistency** implementation
- **Periodic boundary handling** in custom forces
- **Integration with existing potentials** (LJ + custom)
- **Visualization and analysis** of confined systems

## Physics Application

- Models **external confinement** of particle systems
- Useful for **nanoparticle trapping**, **optical tweezers** simulations
- Demonstrates **competition** between thermal motion and external potentials