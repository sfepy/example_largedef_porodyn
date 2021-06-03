import numpy as nm
from porodyn_engine import incremental_algorithm, fc_fce, mat_fce, def_problem


def define():
    params = {
        'mesh_file': 'rect_16x16.vtk',
        'mat_store_elem': 75,   # element for which material data are stored
        'u_store_node': 272,    # node for which displacement is stored
        'p_store_node': 144,    # node for which pressure is stored
        'dim': 2,               # problem dimension
        'dt': 0.01,             # time step
        't_end': 2.0,           # end time
        'force': 4e6,           # applied force
        'save_step': True,      # save results in each time step?
        'init_mode': False,     # calculate initial state?
    }

    material_params = {
        'param': {
            'B': nm.eye(params['dim']),
            'g': 9.81,      # gravitational acceleration
        },
        'solid': {
            'Phi': 0.58,    # volume fraction
            'lam': 8.4e6,   # Lame coefficient
            'mu': 5.6e6,    # Lame coefficient
            'rho': 2700,    # density
        },
        'fluid': {
            'kappa': 1e-1,  # permeability parameter
            'beta': 0.8,    # permeability parameter
            'rho': 1000,    # density
            'Kf': 2.2e10,   # bulk modulus
        },
    }

    regions = {
        'Omega': 'all',
        'Left': ('vertices in (x < 0.001)', 'facet'),
        'Right': ('vertices in (x > 9.999)', 'facet'),
        'Bottom': ('vertices in (y < 0.001)', 'facet'),
        'Top_r': ('vertices in (y > 9.999) & (x > 4.999)', 'facet'),
        'Top_l': ('vertices in (y > 9.999) & (x < 5.001)', 'facet'),
        'ForceRegion': ('copy r.Top_r', 'facet'),
    }

    ebcs = {
        'Fixed_Left_u': ('Left', {'u.0': 0.0}),
        'Fixed_Right_u': ('Right', {'u.0': 0.0}),
        'Fixed_Bottom_u': ('Bottom', {'u.1': 0.0}),
        'Fixed_Top_p': ('Top_l', {'p.0': 0.0}),
    }

    ###############################################

    options = {
        'output_dir': 'output',
        'parametric_hook': 'incremental_algorithm',
    }

    filename_mesh = params['mesh_file']

    materials, functions, fields, variables, equations, solvers = \
        def_problem(params['dt'], params['force'])

    return locals()
