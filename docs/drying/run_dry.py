"""
Run script for drying simulations.
All parameters are defined here and passed to dry.py functions.
"""

import dry
import importlib
importlib.reload(dry)  # Reload if you're editing dry.py
import matplotlib.pyplot as plt
import numpy as np
import hoomd

# ============================================================
# SIMULATION PARAMETERS (CHANGE THESE!)
# ============================================================

# System parameters
N_particles = 1000      # Number of particles
density = 0.002          # σ⁻³
kT = 1.0                # ε/kB

# Dynamics parameters
dt = 0.0001             # Time step
gamma = 1.0             # Translational friction
gamma_r = 1.0           # Rotational friction

# Activity parameters
activity_fraction = 0.0  # Fraction of active particles
force_magnitude = 10.0    # Active force F₀ (ε/σ)

# Wall parameters
wall_epsilon = 1.0       # Wall strength (ε)
wall_r_cut = 2**(1/6)    # Wall cutoff (σ)
Lz_factor = 2.0          # Box aspect ratio 

# Run parameters
n_steps = 100000         # Total simulation steps

# ============================================================
# ANALYSIS PARAMETERS
# ============================================================

# Energy calculation (must match simulation!)
lj_epsilon = 1.0         # Particle-particle LJ epsilon
lj_sigma = 1.0           # Particle-particle LJ sigma
lj_r_cut = 2.5           # Particle-particle LJ cutoff

wall_sigma = 1.0         # Wall sigma (always 1.0)

# Analysis settings
nbins = 100              # Number of spatial bins
skip_frames = 1000        # Frames to skip for equilibration

trajectory_filename = "trajectory.gsd"

# ============================================================
# GET MPI RANK
# ============================================================

try:
    device = hoomd.device.CPU()
    rank = device.communicator.rank
    num_ranks = device.communicator.num_ranks
except:
    rank = 0
    num_ranks = 1

# ============================================================
# RUN SIMULATION
# ============================================================

if rank == 0:
    print("\n" + "="*60)
    print("STARTING DRYING SIMULATION")
    print("="*60 + "\n")


sim = dry.run_simulation(
    N_particles=N_particles,
    density=density,
    dt=dt,
    force_magnitude=force_magnitude,
    gamma_r=gamma_r,
    gamma=gamma,
    kT=kT,
    activity_fraction=activity_fraction,
    wall_epsilon=wall_epsilon,
    wall_r_cut=wall_r_cut,
    n_steps=n_steps,
    Lz_factor=Lz_factor,
)


# ============================================================
# ANALYZE TRAJECTORY (ONLY RANK 0)
# ============================================================

if rank == 0:
    
    
    # Compute thermal susceptibility
    z, chi_T_N_KE, chi_T_N_PE = dry.measure_chi_T(
        trajectory_filename=trajectory_filename,
        kT=kT,
        lj_epsilon=lj_epsilon,
        lj_sigma=lj_sigma,
        lj_r_cut=lj_r_cut,
        wall_epsilon=wall_epsilon,
        wall_sigma=wall_sigma,
        wall_r_cut=wall_r_cut,
        nbins=nbins,
        skip_frames=skip_frames
    )

    # Compute density profile
    z_rho, rho = dry.compute_density_profile(
        trajectory_filename=trajectory_filename,
        nbins=nbins,
        skip_frames=skip_frames
    )
    


    # ============================================================
    # PLOTTING
    # ============================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Density profile
    ax1.plot(z_rho, rho, 'k-', linewidth=2, label='ρ(z)')
    ax1.axhline(y=np.mean(rho[40:60]), color='gray', linestyle='--', alpha=0.5, label='Bulk density')
    ax1.set_xlabel('z/σ', fontsize=12)
    ax1.set_ylabel('ρ(z) [σ⁻³]', fontsize=12)
    ax1.set_title('Density Profile', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # # Thermal susceptibility
# Thermal susceptibility - now plot both KE and PE
    ax2.plot(z, chi_T_N_KE, 'b-', linewidth=2, label='χ_T,N(z) - KE', alpha=0.7)
    ax2.plot(z, chi_T_N_PE, 'r-', linewidth=2, label='χ_T,N(z) - PE', alpha=0.7)
    ax2.plot(z, chi_T_N_KE + chi_T_N_PE, 'k-', linewidth=2, label='χ_T,N(z) - Total')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    chi_bulk_total = np.mean(chi_T_N_KE[40:60] + chi_T_N_PE[40:60])
    ax2.axhline(y=chi_bulk_total, color='gray', linestyle='--', alpha=0.5, label='Bulk total')
    ax2.set_xlabel('z/σ', fontsize=12)
    ax2.set_ylabel('χ_T,N(z) [σ³]', fontsize=12)
    ax2.set_title('Thermal Susceptibility', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fluctuation_profiles.png', dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved as 'fluctuation_profiles.png'\n")
    plt.show()

    # ============================================================
    # SUMMARY
    # ============================================================

    chi_T_N_total = chi_T_N_KE + chi_T_N_PE
    chi_bulk = np.mean(chi_T_N_total[40:60])
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Contact density:      ρ(0) = {rho[nbins//2]:.4f} σ⁻³")
    print(f"Bulk density:         ρ_∞  = {np.mean(rho[40:60]):.4f} σ⁻³")
    print(f"Contact χ_T,N:        χ(0) = {chi_T_N_total[nbins//2]:.4f} σ³")
    print(f"Bulk χ_T,N:           χ_∞  = {chi_bulk:.4f} σ³")
    print(f"Enhancement factor:   χ(0)/χ_∞ = {chi_T_N_total[nbins//2]/chi_bulk:.2f}")
    print("="*60)
    
