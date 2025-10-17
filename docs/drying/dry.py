def create_simulation(N_particles, density, dt, force_magnitude, gamma_r, gamma, kT, 
                     activity_fraction, wall_epsilon, wall_r_cut, Lz_factor):
    """
    Create HOOMD-blue simulation with walls.
    """
    import hoomd
    import numpy as np
    import gsd.hoomd
    import os
    
    box_volume = N_particles / density
    L = box_volume**(1/3)

    Lx = L * Lz_factor**-0.5
    Ly = L * Lz_factor**-0.5
    Lz = L * Lz_factor


    # Create device first to get communicator
    cpu = hoomd.device.CPU()
    rank = cpu.communicator.rank
    
    # Only rank 0 creates the initial configuration file
    init_filename = "init.gsd"
    
    if rank == 0:
        frame = gsd.hoomd.Frame()
        frame.particles.N = N_particles
        # FCC lattice placement
        margin = 0.5
        np.random.seed(None)

        # FCC has 4 atoms per unit cell
        atoms_per_cell = 4
        n_cells = int(np.ceil((N_particles / atoms_per_cell)**(1/3)))

        # Lattice constant
        a_x = Lx / n_cells
        a_y = Ly / n_cells
        a_z = (Lz - 2*margin) / n_cells

        # FCC basis vectors (in units of lattice constant)
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])

        positions = []
        for i in range(n_cells):
            for j in range(n_cells):
                for k in range(n_cells):
                    for b in basis:
                        if len(positions) >= N_particles:
                            break
                        x = -Lx/2 + (i + b[0]) * a_x
                        y = -Ly/2 + (j + b[1]) * a_y
                        z = -Lz/2 + margin + (k + b[2]) * a_z
                        positions.append([x, y, z])

        positions = np.array(positions[:N_particles])

        frame.particles.position = positions
        
        # Assign particle types
        N_active = int(N_particles * activity_fraction)
        typeid = [0] * N_active + [1] * (N_particles - N_active)
        np.random.shuffle(typeid)
        frame.particles.typeid = typeid 
        frame.particles.types = ["A", "P"]
        frame.particles.moment_inertia = [[1.0, 1.0, 1.0]] * N_particles
        frame.configuration.box = [Lx, Ly, Lz, 0, 0, 0]
        
        # Random orientations
        np.random.seed(None)
        random_orientations = []
        for i in range(N_particles):
            q = np.random.randn(4)
            q = q / np.linalg.norm(q)
            random_orientations.append(q)
        
        frame.particles.orientation = random_orientations

        # Create temporary initial configuration file (only rank 0)
        if os.path.exists(init_filename):
            os.remove(init_filename)

        with gsd.hoomd.open(name=init_filename, mode="x") as f:
            f.append(frame)
        
        print(f"Running on {cpu.communicator.num_ranks} MPI ranks")
    
    # Barrier: wait for rank 0 to finish creating the file
    cpu.communicator.barrier()
    
    simulation = hoomd.Simulation(device=cpu, seed=1)
    simulation.create_state_from_gsd(filename=init_filename)

    # Print decomposition details (only rank 0)
    if rank == 0:
        domain_decomp = simulation.state.domain_decomposition
        print(f"Domain decomposition: {domain_decomp}")
    
    simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)
    
    # Barrier before cleanup
    cpu.communicator.barrier()
    
    # Clean up init file (only rank 0)
    if rank == 0 and os.path.exists(init_filename):
        os.remove(init_filename)
    
    print('HERE')
    integrator = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=False)

    # Active force
    if force_magnitude > 0:
        active_filter = hoomd.filter.Type(['A'])
        active_force = hoomd.md.force.Active(filter=active_filter)
        active_force.active_force['A'] = (force_magnitude, 0.0, 0.0)
        active_force.active_torque['A'] = (0.0, 0.0, 0.0)
        integrator.forces.append(active_force)
    
    # Brownian dynamics
    brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=kT)  
    brownian.gamma['A'] = gamma  
    brownian.gamma['P'] = gamma
    brownian.gamma_r['A'] = (gamma_r, gamma_r, gamma_r)
    brownian.gamma_r['P'] = (gamma/3, gamma/3, gamma/3)
    integrator.methods.append(brownian)

    # Neighbor list
    cell = hoomd.md.nlist.Cell(buffer=0.019)

    # Lennard-Jones potential
    lj = hoomd.md.pair.LJ(nlist=cell, mode='shift')
    lj.params[("A", "A")] = dict(epsilon=1, sigma=1)
    lj.params[("A", "P")] = dict(epsilon=1, sigma=1) 
    lj.params[("P", "P")] = dict(epsilon=1, sigma=1)
    lj.r_cut[("A", "A")] = 2.5
    lj.r_cut[("A", "P")] = 2.5  
    lj.r_cut[("P", "P")] = 2.5
    integrator.forces.append(lj)
    
    # Walls
    lj_walls = [
        hoomd.wall.Plane(origin=(0.0, 0.0, -Lz/2), normal=(0.0, 0.0, 1.0)),
        hoomd.wall.Plane(origin=(0.0, 0.0, Lz/2), normal=(0.0, 0.0, -1.0))
    ]
    shifted_lj_wall = hoomd.md.external.wall.ForceShiftedLJ(walls=lj_walls)
    shifted_lj_wall.params["A"] = {
        "epsilon": wall_epsilon,
        "sigma": 1.0,
        "r_cut": wall_r_cut,
    }
    shifted_lj_wall.params["P"] = {
        "epsilon": wall_epsilon,
        "sigma": 1.0,
        "r_cut": wall_r_cut,
    }
    integrator.forces.append(shifted_lj_wall)
    
    constant_volume = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    fire = hoomd.md.minimize.FIRE(
        dt=0.0001,
        force_tol=1e-8,
        angmom_tol=1e-8,
        energy_tol=1e-8,
        methods=[constant_volume],
    )
    simulation.operations.integrator = fire
    print("Minimizing energy...")
    while not fire.converged:
        simulation.run(100)
    print("✓ Energy minimized!\n")
    simulation.operations.integrator = integrator

    # Delete old trajectory file if it exists (only rank 0)
    trajectory_filename = "trajectory.gsd"
    if rank == 0 and os.path.exists(trajectory_filename):
        os.remove(trajectory_filename)
    
    # Barrier to ensure file is deleted before writers are created
    cpu.communicator.barrier()
    
    gsd_writer = hoomd.write.GSD(
        filename=trajectory_filename, 
        trigger=hoomd.trigger.Periodic(period=100),
        mode="wb",
        filter=hoomd.filter.All(),
        dynamic=['property', 'momentum']
    )
    
    # Log energies to the GSD file
    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(lj, quantities=['energy'])
    logger.add(shifted_lj_wall, quantities=['energy'])
    gsd_writer.logger = logger
    
    simulation.operations.writers.append(gsd_writer)
    
    return simulation

def run_simulation(N_particles, density, dt, force_magnitude, gamma_r, gamma, kT, 
                   activity_fraction, wall_epsilon, wall_r_cut, n_steps, Lz_factor):
    import hoomd
    import time
    import sys

    """
    Run the simulation with progress tracking via custom writer.
    
    Parameters:
    -----------
    (Same as create_simulation, plus:)
    n_steps : int
        Number of simulation steps to run
    
    Returns:
    --------
    simulation : hoomd.Simulation
        The completed simulation object
    elapsed_time : float
        Wall time for simulation run in seconds
    """
    progress_interval = n_steps // 10000

    simulation = create_simulation(
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
        Lz_factor=Lz_factor
    )
    
    device = simulation.device
    rank = device.communicator.rank

    if rank == 0:
        print("="*60)
        print("SIMULATION SETUP")
        print("="*60)
        print(f"  N particles:        {N_particles}")
        print(f"  Density:            {density:.3f} σ⁻³")
        print(f"  Temperature:        {kT:.3f} ε/kB")
        print(f"  Timestep:           {dt:.5f} τ")
        print(f"  Activity fraction:  {activity_fraction:.2f}")
        print(f"  Active force:       {force_magnitude:.2f} ε/σ")
        print(f"  Wall epsilon:       {wall_epsilon:.2f} ε")
        print(f"  Wall cutoff:        {wall_r_cut:.3f} σ")
        print(f"  Box aspect ratio:   {Lz_factor:.2f}")
        print(f"  Total steps:        {n_steps}")
        print("="*60)
        print("\nRunning simulation...")
    
    # Add progress tracker if requested
    if progress_interval > 0 and rank == 0:
        
        # Try to open terminal directly to bypass buffering
        try:
            tty = open('/dev/tty', 'w')
        except:
            tty = sys.stdout
        
        class ProgressWriter(hoomd.custom.Action):
            """Custom action to print progress."""
            def __init__(self, total_steps, start_time, output_stream):
                self.total_steps = total_steps
                self.start_time = start_time
                self.output = output_stream
            
            def act(self, timestep):
                percent = (timestep / self.total_steps) * 100
                elapsed = time.time() - self.start_time
                
                if elapsed > 0 and timestep > 0:
                    steps_per_sec = timestep / elapsed
                    remaining = (self.total_steps - timestep) / steps_per_sec
                    eta_min = remaining / 60
                    
                    self.output.write(f"\r  Progress: {percent:.1f}% ({timestep}/{self.total_steps}) | "
                          f"{steps_per_sec:.0f} steps/s | ETA: {eta_min:.1f} min")
                    self.output.flush()
        
        progress_writer = hoomd.write.CustomWriter(
            action=ProgressWriter(n_steps, time.time(), tty),
            trigger=hoomd.trigger.Periodic(progress_interval)
        )
        simulation.operations.writers.append(progress_writer)
    
    # Start timer for simulation run only
    start_time = time.time()
    
    # Run simulation in one go (no chunking!)
    simulation.run(n_steps)
    
    # End timer
    elapsed_time = time.time() - start_time
    
    if rank == 0:
        print(f"\n\n✓ Simulation completed! {n_steps} steps done.\n")
        
        # Print performance statistics
        print("="*60)
        print("SIMULATION PERFORMANCE")
        print("="*60)
        print(f"Wall time:            {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Performance:          {simulation.tps:.2f} timesteps/second")
        print(f"MPI ranks used:       {device.communicator.num_ranks}")
        print(f"CPU time:             {elapsed_time * device.communicator.num_ranks:.2f} CPU-seconds")
        if simulation.tps > 0:
            print(f"Time for 1M steps:    {1e6 / simulation.tps / 60:.2f} minutes")
        print("="*60 + "\n")
    
    # Flush the trajectory writer to ensure all data is written to disk
    for writer in simulation.operations.writers:
        if hasattr(writer, 'flush'):
            writer.flush()
    
    return simulation, elapsed_time
def lj_potential(r, epsilon, sigma):
    """Standard Lennard-Jones potential."""
    r6_inv = (sigma / r)**6
    r12_inv = r6_inv**2
    return 4 * epsilon * (r12_inv - r6_inv)


def compute_frame_energy(frame, lj_epsilon, lj_sigma, lj_r_cut, 
                         wall_epsilon, wall_sigma, wall_r_cut):
    """
    Compute total potential energy for a single frame.
    
    Parameters:
    -----------
    frame : gsd.hoomd.Frame
        Single trajectory frame
    lj_epsilon : float
        LJ energy parameter (particle-particle)
    lj_sigma : float
        LJ length parameter (particle-particle)
    lj_r_cut : float
        LJ cutoff distance (particle-particle)
    wall_epsilon : float
        Wall energy parameter
    wall_sigma : float
        Wall length parameter
    wall_r_cut : float
        Wall cutoff distance
    
    Returns:
    --------
    total_energy : float
        Total potential energy (LJ + wall)
    """
    import numpy as np
    
    positions = frame.particles.position
    N = len(positions)
    box_size = frame.configuration.box[:3]
    L = box_size[2]  # z-dimension
    
    # 1. Particle-particle LJ energy
    U_lj = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            dr = positions[j] - positions[i]
            
            # Minimum image convention (periodic in x, y)
            dr[0] -= box_size[0] * np.round(dr[0] / box_size[0])
            dr[1] -= box_size[1] * np.round(dr[1] / box_size[1])
            
            r = np.linalg.norm(dr)
            
            if r < lj_r_cut:
                # Shifted LJ: U(r) - U(r_cut)
                U_lj += lj_potential(r, lj_epsilon, lj_sigma) - lj_potential(lj_r_cut, lj_epsilon, lj_sigma)
    
    # 2. Wall potential
    U_wall = 0.0
    
    # For WCA (r_cut = 2^(1/6)), the shift makes U(r_cut) = 0
    # U_WCA(z) = U_LJ(z) + epsilon
    if wall_r_cut == 2**(1/6) * wall_sigma:
        U_shift = wall_epsilon  # WCA shift
    else:
        # For general cutoff, shift to make U(r_cut) = 0
        U_shift = -lj_potential(wall_r_cut, wall_epsilon, wall_sigma)
    
    wall_positions = [-L/2, L/2]
    wall_normals = [1.0, -1.0]
    
    for wall_z, normal in zip(wall_positions, wall_normals):
        for particle_pos in positions:
            z_dist = normal * (particle_pos[2] - wall_z)
            
            if 0 < z_dist < wall_r_cut:
                U_lj_wall = lj_potential(z_dist, wall_epsilon, wall_sigma)
                U_wall += U_lj_wall + U_shift
    
    return U_lj + U_wall


def measure_chi_T(trajectory_filename, kT, lj_epsilon, lj_sigma, lj_r_cut,
                  wall_epsilon, wall_sigma, wall_r_cut, nbins=100, skip_frames=100):
    """
    Compute canonical thermal susceptibility χ_T,N(z) from trajectory.
    Now returns separate susceptibilities for KE and PE.
    
    Returns:
    --------
    z_centers : np.ndarray
        Z-coordinates of bin centers
    chi_T_N_KE : np.ndarray
        Thermal susceptibility profile for kinetic energy
    chi_T_N_PE : np.ndarray
        Thermal susceptibility profile for potential energy
    """
    import gsd.hoomd
    import numpy as np
    
    print("="*60)
    print("THERMAL SUSCEPTIBILITY ANALYSIS")
    print("="*60)
    print(f"Reading trajectory from {trajectory_filename}...")
    
    with gsd.hoomd.open(trajectory_filename, mode='r') as traj:
        total_frames = len(traj)
        
        # Auto-adjust skip_frames if needed
        if skip_frames >= total_frames:
            print(f"WARNING: skip_frames ({skip_frames}) >= total frames ({total_frames})")
            skip_frames = max(1, total_frames // 10)
            print(f"         Adjusted to {skip_frames}")
        
        if total_frames - skip_frames < 10:
            raise ValueError(
                f"Not enough frames! Total: {total_frames}, Skip: {skip_frames}, "
                f"Remaining: {total_frames - skip_frames}. Need at least 10 frames."
            )
        
        print(f"  Total frames:   {total_frames}")
        print(f"  Skip frames:    {skip_frames}")
        print(f"  Using frames:   {total_frames - skip_frames}")
        print(f"  Spatial bins:   {nbins}")
        
        # Get box dimensions
        L = traj[0].configuration.box[2]
        
        # Set up spatial bins
        z_bins = np.linspace(-L/2, L/2, nbins + 1)
        z_centers = 0.5 * (z_bins[1:] + z_bins[:-1])
        
        # Storage for time series
        kinetic_energies = []
        potential_energies = []
        density_profiles = []
        
        print("Processing frames...")
        
        # Loop over trajectory (skip equilibration)
        for frame_idx, frame in enumerate(traj[skip_frames:]):
            if frame_idx % 100 == 0:
                print(f"  Progress: {frame_idx}/{total_frames - skip_frames}", end='\r')
            
            # Potential energy from logged data
            PE = frame.log['md/pair/LJ/energy'][0] + frame.log['md/external/wall/ForceShiftedLJ/energy'][0]
            potential_energies.append(PE)
            
            # Kinetic energy from velocities
            velocities = frame.particles.velocity
            masses = np.ones(len(velocities))  # Assuming unit mass
            KE = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
            kinetic_energies.append(KE)
            
            # Bin particles by z-position
            z_positions = frame.particles.position[:, 2]
            hist, _ = np.histogram(z_positions, bins=z_bins)
            density_profiles.append(hist)
        
        print(f"  Progress: {total_frames - skip_frames}/{total_frames - skip_frames}")
        
        # Convert to arrays
        kinetic_energies = np.array(kinetic_energies)
        potential_energies = np.array(potential_energies)
        density_profiles = np.array(density_profiles)
        
        if len(kinetic_energies) == 0:
            raise ValueError("No frames were processed!")
        
        print("Computing covariances...")
        
        # Compute covariance for each spatial bin - separately for KE and PE
        chi_T_N_KE = np.zeros(nbins)
        chi_T_N_PE = np.zeros(nbins)
        
        for i in range(nbins):
            cov_KE = np.cov(kinetic_energies, density_profiles[:, i])[0, 1]
            cov_PE = np.cov(potential_energies, density_profiles[:, i])[0, 1]
            chi_T_N_KE[i] = cov_KE / (kT**2)
            chi_T_N_PE[i] = cov_PE / (kT**2)
        
        # Diagnostics
        print("\nDiagnostics:")
        print(f"  Mean KE:        {np.mean(kinetic_energies):.2f} ε")
        print(f"  KE std:         {np.std(kinetic_energies):.2f} ε")
        print(f"  Mean PE:        {np.mean(potential_energies):.2f} ε")
        print(f"  PE std:         {np.std(potential_energies):.2f} ε")
        print(f"  χ_T,N(KE) range: [{np.min(chi_T_N_KE):.4f}, {np.max(chi_T_N_KE):.4f}] σ³")
        print(f"  χ_T,N(PE) range: [{np.min(chi_T_N_PE):.4f}, {np.max(chi_T_N_PE):.4f}] σ³")
        print(f"  ∫χ_T,N(KE) dz:  {np.trapz(chi_T_N_KE, z_centers):.4f} σ²")
        print(f"  ∫χ_T,N(PE) dz:  {np.trapz(chi_T_N_PE, z_centers):.4f} σ²")
        print("="*60)
        print("✓ Analysis complete!\n")
        
    return z_centers, chi_T_N_KE, chi_T_N_PE


def compute_density_profile(trajectory_filename, nbins=100, skip_frames=100):
    """
    Compute density profile ρ(z) from trajectory.
    
    Parameters:
    -----------
    trajectory_filename : str
        Path to GSD trajectory file
    nbins : int
        Number of spatial bins
    skip_frames : int
        Frames to skip for equilibration
    
    Returns:
    --------
    z_centers : np.ndarray
        Z-coordinates of bin centers
    rho : np.ndarray
        Density profile
    """
    import gsd.hoomd
    import numpy as np
    
    with gsd.hoomd.open(trajectory_filename, mode='r') as traj:
        total_frames = len(traj)
        
        if skip_frames >= total_frames:
            skip_frames = max(1, total_frames // 10)
        
        L = traj[0].configuration.box[2]
        z_bins = np.linspace(-L/2, L/2, nbins + 1)
        z_centers = 0.5 * (z_bins[1:] + z_bins[:-1])
        
        # Bin volume
        Lx, Ly = traj[0].configuration.box[0], traj[0].configuration.box[1]
        bin_volume = (L / nbins) * Lx * Ly
        
        rho = np.zeros(nbins)
        
        for frame in traj[skip_frames:]:
            z_positions = frame.particles.position[:, 2]
            hist, _ = np.histogram(z_positions, bins=z_bins)
            rho += hist
        
        # Normalize
        rho /= (total_frames - skip_frames) * bin_volume
        
    return z_centers, rho