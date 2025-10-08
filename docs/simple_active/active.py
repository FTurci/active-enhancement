def create_simulation(N_particles, density, dt, force_magnitude, rotational_diffusion, gamma, kT, activity_fraction):
    import hoomd
    import numpy as np
    import math
    import itertools
    import gsd.hoomd
    import os
    box_volume = N_particles/density
    L = box_volume**(1/3)


    x = np.linspace(-L / 2 + 0.5, L / 2 - 0.5, math.ceil(N_particles ** (1/3)), endpoint=False)
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
    
    
    Brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=kT)  
    

    Brownian.gamma['A'] = gamma  
    Brownian.gamma['P'] = gamma  
    # Rotational diffusion as desired
    Brownian.gamma_r['A'] = (rotational_diffusion, rotational_diffusion, rotational_diffusion)
    Brownian.gamma_r['P'] = (rotational_diffusion, rotational_diffusion, rotational_diffusion)

    integrator.methods.append(Brownian)

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
    
    # Clean up
    if os.path.exists(filename):
        os.remove(filename)


    # Switch back to regular integrator
    simulation.operations.integrator = integrator


    gsd_writer = hoomd.write.GSD(filename="trajectory.gsd", 
                            trigger=hoomd.trigger.Periodic(period=100),
                            mode="wb",
                            filter=hoomd.filter.All(),
                            dynamic=['property', 'momentum'])
    simulation.operations.writers.append(gsd_writer)

    return simulation

# Cell 3
# --------------------------------------------------
def run_simulation(steps, N_particles, density, dt, force_magnitude, rotational_diffusion, gamma, kT, activity_fraction):
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
                            gamma,
                            activity_fraction
                            )
    
    simulation.run(steps)
    print("Simulation completed!")
