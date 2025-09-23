# Converted from dry.ipynb
# Generated automatically - do not edit directly

# Cell 1
# --------------------------------------------------
# Cell 2
# --------------------------------------------------
def create_simulation(N_particles, density, dt, force_magnitude, rotational_diffusion, kT, activity_fraction):
    import hoomd
    import numpy as np
    import math
    import itertools
    import gsd.hoomd
    import os
    box_volume = N_particles/density
    L = box_volume**(1/3)


    x = np.linspace(-L / 2, L / 2, math.ceil(N_particles ** (1/3)), endpoint=False)
    position = list(itertools.product(x, repeat=3))
    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = position[0:N_particles]
    N_active = int(N_particles * activity_fraction)
    typeid = [0] * N_active + [1] * (N_particles - N_active)
    np.random.shuffle(typeid)  # Randomly mix active and passive particles
    frame.particles.typeid = typeid 
    frame.particles.types = ["A", "P"]  # A = active, P = passive
    frame.particles.moment_inertia = [[1.0, 1.0, 1.0]] * N_particles  # Enable rotation
    frame.configuration.box = [L, L, L, 0, 0, 0]
        # Add random starting orientations
    np.random.seed(42)  # For reproducible results, remove if you want truly random
    random_orientations = []
    for i in range(N_particles):
        # Generate random quaternion by normalizing 4 random numbers
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)  # Normalize to unit quaternion
        random_orientations.append(q)
    
    frame.particles.orientation = random_orientations

    filename="trajectory.gsd"
    if os.path.exists(filename):
        os.remove(filename)

    # Now open and write to the file
    with gsd.hoomd.open(name=filename, mode="x") as f:
        f.append(frame)
    
    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device=cpu, seed=1)
    simulation.create_state_from_gsd(filename=filename)

        # Create the integrator
    integrator = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=True)

    if force_magnitude > 0:
        active_filter = hoomd.filter.Type(['A'])
        active_force = hoomd.md.force.Active(filter=active_filter)
        active_force.active_force['A'] = (force_magnitude, 0.0, 0.0)
        active_force.active_torque['A'] = (0.0, 0.0, 0.0)
        integrator.forces.append(active_force)
    
    
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=kT)  
    

    langevin.gamma['A'] = 1  
    langevin.gamma['P'] = 1  
    # Rotational diffusion as desired
    langevin.gamma_r['A'] = (rotational_diffusion, rotational_diffusion, rotational_diffusion)
    langevin.gamma_r['P'] = (rotational_diffusion, rotational_diffusion, rotational_diffusion)

    integrator.methods.append(langevin)

    # Set up the neighbor list
    cell = hoomd.md.nlist.Cell(buffer=0.019)  

    # Set up Lennard-Jones potential  
    lj = hoomd.md.pair.LJ(nlist=cell, mode='shift')
    lj.params[("A", "A")] = dict(epsilon=1, sigma=1)
    lj.params[("A", "P")] = dict(epsilon=1, sigma=1) 
    lj.params[("P", "P")] = dict(epsilon=1, sigma=1)
    lj.r_cut[("A", "A")] = 2
    lj.r_cut[("A", "P")] = 2  
    lj.r_cut[("P", "P")] = 2 

    

    # Add the force to the integrator
    integrator.forces.append(lj)

    # Assign the integrator to the simulation
    simulation.operations.integrator = integrator

    equilibration_steps = 10000  # Let system reach equilibrium
    simulation.run(equilibration_steps)
    equilibration_timestep = simulation.timestep
    # Store positions starting from equilibrated state
    positions = []
    times = []
    
    # Get initial position after equilibration
    snapshot = simulation.state.get_snapshot()
    initial_position = snapshot.particles.position[0].copy()
    positions.append(initial_position)
    times.append(simulation.timestep - equilibration_timestep)  # Reset time reference
    
    # Clean up
    if os.path.exists(filename):
        os.remove(filename)
    
    lj_walls = [hoomd.wall.Plane(origin=(0.0, 0.0, -L/2-0.5), normal=(0.0, 0.0, 1.0)),
            hoomd.wall.Plane(origin=(0.0, 0.0, L/2+0.5), normal=(0.0, 0.0, -1.0))]
    shifted_lj_wall = hoomd.md.external.wall.ForceShiftedLJ(walls=lj_walls)
    shifted_lj_wall.params["A"] = {
        "epsilon": 1.0,
        "sigma": 1.0,
        "r_cut": 2**(1/6),
    }
    shifted_lj_wall.params["P"] = {
    "epsilon": 1.0,
    "sigma": 1.0,
    "r_cut": 2**(1/6),
    }

    integrator.forces.append(shifted_lj_wall)


    gsd_writer = hoomd.write.GSD(filename="trajectory.gsd", 
                            trigger=hoomd.trigger.Periodic(period=100),
                            mode="wb")
    simulation.operations.writers.append(gsd_writer)

    return simulation

# Cell 3
# --------------------------------------------------
def run_simulation(N_particles, density, dt, force_magnitude, rotational_diffusion, kT, activity_fraction):
    import hoomd
    import numpy as np
    import math
    import itertools
    import gsd.hoomd
    import os
    
    """Run the simulation."""
    print("Setting up simulation...")
    simulation = create_simulation(N_particles, 
                            density,  
                            dt, 
                            force_magnitude, 
                            rotational_diffusion, 
                            kT,
                            activity_fraction
                            )
    
    simulation.run(100000)
    print("Simulation completed!")

# Cell 4
# --------------------------------------------------
if __name__ == "__main__":
    run_simulation(N_particles=1000, 
                            density=0.1,  
                            dt=0.0001, 
                            force_magnitude=5, 
                            rotational_diffusion=0.1, 
                            kT=1,
                            activity_fraction=0.5
                            )

# Cell 5
# --------------------------------------------------
def analyze_density_profile(trajectory_file, bins):
    import hoomd
    import numpy as np
    import math
    import itertools
    import gsd.hoomd
    import os
    import freud
    """Analyze density profiles from saved trajectory."""
    densities_z = []
    
    with gsd.hoomd.open(trajectory_file, 'r') as traj:
        for frame in traj:
            box = frame.configuration.box
            positions = frame.particles.position
            
            # Create z-density profile
            z_coords = positions[:, 2]
            hist, bin_edges = np.histogram(z_coords, 
                                         bins=bins, 
                                         range=(-box[2]/2, box[2]/2))
            
            # Convert to density
            bin_width = bin_edges[1] - bin_edges[0]
            bin_volume = box[0] * box[1] * bin_width  # Assuming uniform xy
            density = hist / bin_volume
            
            densities_z.append(density)
    
    densities, z_bins = np.array(densities_z), bin_edges
    average_density = np.mean(densities, axis=0)

    # Plot
    import matplotlib.pyplot as plt
    z_centers = (z_bins[1:] + z_bins[:-1]) / 2
    plt.plot(z_centers, average_density)
    plt.xlabel('Z position')
    plt.ylabel('Density')
    plt.show()

# Cell 6
# --------------------------------------------------
def analyze_seperate_profiles(trajectory_file, bins, active_fraction):
    import freud
    import hoomd
    import numpy as np
    import math
    import itertools
    import gsd.hoomd
    import os
    """Analyze density profiles from saved trajectory, separated by particle type."""
    densities_active = []
    densities_passive = []
    
    with gsd.hoomd.open(trajectory_file, 'r') as traj:
        for frame in traj:
            box = frame.configuration.box
            positions = frame.particles.position
            typeids = frame.particles.typeid
            
            # Separate positions by type (0 = Active, 1 = Passive)
            active_positions = positions[typeids == 0]
            passive_positions = positions[typeids == 1]
            
            # Create z-density profiles for each type
            z_range = (-box[2]/2, box[2]/2)
            bin_width = (z_range[1] - z_range[0]) / bins
            bin_volume = box[0] * box[1] * bin_width
            
            # Define bin_edges here so it's always available
            _, bin_edges = np.histogram([], bins=bins, range=z_range)
            
            # Active particles
            if len(active_positions) > 0:
                z_coords_active = active_positions[:, 2]
                hist_active, _ = np.histogram(z_coords_active, bins=bins, range=z_range)
                density_active = hist_active / bin_volume
            else:
                density_active = np.zeros(bins)
            
            # Passive particles  
            if len(passive_positions) > 0:
                z_coords_passive = passive_positions[:, 2]
                hist_passive, _ = np.histogram(z_coords_passive, bins=bins, range=z_range)
                density_passive = hist_passive / bin_volume
            else:
                density_passive = np.zeros(bins)
            
            densities_active.append(density_active)
            densities_passive.append(density_passive)
    
    densities_active, densities_passive, z_bins = np.array(densities_active), np.array(densities_passive), bin_edges
    average_density_active = np.mean(densities_active, axis=0)
    average_density_passive = np.mean(densities_passive, axis=0)

    # Plot both on same graph
    import matplotlib.pyplot as plt
    z_centres = (z_bins[1:] + z_bins[:-1]) / 2
    plt.plot(z_centres, average_density_active, label='Active particles', color='red')
    plt.plot(z_centres, average_density_passive, label='Passive particles', color='blue')
    plt.plot(z_centres, average_density_passive+average_density_active, label='Active + Passive ', color='black')
    plt.xlabel('Z position')
    plt.ylabel('Density')
    plt.title(f'Active fraction = {active_fraction:.3f}')
    plt.legend()
    plt.savefig(f'plots/AF={active_fraction:.3f}.png')
    plt.show()

# Cell 7
# --------------------------------------------------
# Cell 8
# --------------------------------------------------