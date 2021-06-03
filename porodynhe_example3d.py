# Rohan E., Lukeš V.
# Modeling large-deforming fluid-saturated porous media using
# an Eulerian incremental formulation.
# Advances in Engineering Software, 113:84-95, 2017,
# https://doi.org/10.1016/j.advengsoft.2016.11.003
#
# Run simulation:
#
#   ./simple.py example_largedef_porodyn-1/porodynhe_example3d.py
#
# The results are stored in `example_largedef_porodyn-1/results`.
#

import numpy as nm
from porodyn_engine import incremental_algorithm,\
    fc_fce, mat_fce, def_problem
import os.path as osp

wdir = osp.dirname(__file__)


def define():
    params = {
        'mesh_file': 'cylinder.vtk',
        'dim': 3,               # problem dimension
        'dt': 0.01,             # time step
        't_end': 1.5,           # end time
        # 'force': 12e6,           # applied force
        'displ': 0.02,          # applied displacement
        'save_step': True,      # save results in each time step?
        'init_mode': False,     # calculate initial state?
        'mat_store_elem': 75,   # element for which material data are stored
    }

    material_params = {
        'param': {
            'B': nm.eye(params['dim']),
            'g': 9.81,      # gravitational acceleration
        },
        'solid': {
            'Phi': 0.58,    # volume fraction
            'lam': 29e6,    # Lame coefficient
            'mu': 7e6,      # Lame coefficient
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
        'Surface': ('vertices of surface', 'facet'),
        'Bottom': ('vertices in (z < 0.001)', 'facet'),
        'Top': ('vertices in (z > 0.099)', 'facet'),
        'Side': ('r.Surface -f r.Top -f r.Bottom', 'facet'),
        # 'ForceRegion': ('copy r.Top', 'facet'),
        'MovingRegion': ('copy r.Top', 'facet'),
    }

    ebcs = {
        'Fixed_Side_u': ('Side', {'u.0': 0.0, 'u.1': 0.0}),
        'Fixed_Bottom_u': ('Bottom', {'u.2': 0.0}),
        'Fixed_Top_p': ('Top', {'p.0': 0.0}),
        # 'Moving_Region': ('MovingRegion', {'u.2': 'displ_fce'}),
    }

    ###############################################

    options = {
        'output_dir': osp.join(wdir, 'results'),
        'parametric_hook': 'incremental_algorithm',
    }

    filename_mesh = params['mesh_file']

    forceparam = params['force'] if 'force' in params else None
    displparam = params['displ'] if 'displ' in params else None
    materials, functions, fields, variables, equations, solvers = \
        def_problem(params['dt'], forceparam, displparam)

    return locals()
