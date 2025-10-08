import active
import importlib
importlib.reload(active)

active.run_simulation(100000, #steps
                           100, #N_particles 
                            0.0001,  #density 
                            0.001, #dt
                            10, #force
                            10, #rotational diffustion 
                            1, #gamma
                            1, #kT
                            1 #activity fraction
                            )
