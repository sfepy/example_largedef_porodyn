import numpy as nm
import copy
from sfepy.base.base import Struct, output
from sfepy.linalg.utils import dot_sequences
from sfepy.terms.terms_hyperelastic_ul import HyperElasticULFamilyData
import os.path as osp
from sfepy.base.base import debug

hyperela = {
    'step': 0,
    'step_in_cache': -1,
    'cache': {},
    'state': {'u': None, 'du': None, 'du1': None,
              'p': None, 'dp': None},
    'mapping0': None,
    'coors0': None,
    'mhist': {},
    'aux': {},
    # 'residual_pred': None,
}

sym_eye = {
    2: nm.array([1, 1, 0]).reshape((3, 1)),
    3: nm.array([1, 1, 1, 0, 0, 0]).reshape((6, 1)),
}

ssym_tab = {
    2: [0, 3, 1],
    3: [0, 4, 8, 5, 2, 1],
}

delta_delta = {
    2: nm.array([[2, 0, 0, 0],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 0, 0, 2]]),
    3: nm.array([[2, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 1, 0, 0],
                 [0, 1, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 1, 0],
                 [0, 0, 1, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 2]])
}

one_one = {
    2: nm.array([[1, 0, 0, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [1, 0, 0, 1]]),
    3: nm.array([[1, 0, 0, 0, 1, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 1, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 1, 0, 0, 0, 1]]),
}

stress_Atab = {
    2: [(0, 0, 0), (0, 2, 2), (1, 1, 1), (1, 3, 3),
        (2, 0, 1), (2, 1, 0), (2, 2, 3), (2, 3, 2)],
    3: [(0, 0, 0), (0, 3, 3), (0, 6, 6),
        (1, 1, 1), (1, 4, 4), (1, 7, 7),
        (2, 2, 2), (2, 5, 5), (2, 8, 8),
        (3, 0, 1), (3, 1, 0), (3, 3, 4), (3, 4, 3), (3, 6, 7), (3, 7, 6),
        (4, 0, 2), (4, 2, 0), (4, 3, 5), (4, 5, 3), (4, 6, 8), (4, 8, 6),
        (5, 1, 2), (5, 2, 1), (5, 4, 5), (5, 5, 4), (5, 7, 8), (5, 8, 7)]
}

press_Atab = {
    2: [(2, 1., 0, 1), (2, -1., 1, 0), (2, 1., 3, 2), (2, -1., 2, 3),
        (0, -1., 0, 3), (0, 1., 1, 2), (1, 1., 2, 1), (1, -1., 3, 0)],
    3: [(0, -1., 0, 4), (0, -1., 0, 8), (0, 1., 1, 3), (0, 1., 2, 6),
        (1, 1., 3, 1), (1, -1., 4, 0), (1, -1., 4, 8), (1, 1., 5, 7),
        (2, 1., 6, 2), (2, 1., 7, 5), (2, -1., 8, 0), (2, -1., 8, 4),
        (3, 1., 0, 1), (3, -1., 1, 0), (3, -1., 1, 8), (3, 1., 2, 7),
        (3, -1., 3, 4), (3, -1., 3, 8), (3, 1., 4, 3), (3, 1., 5, 6),
        (4, 1., 0, 2), (4, 1., 1, 5), (4, -1., 2, 0), (4, -1., 2, 4),
        (4, -1., 6, 4), (4, -1., 6, 8), (4, 1., 7, 3), (4, 1., 8, 6),
        (5, 1., 3, 2), (5, 1., 4, 5), (5, -1., 5, 0), (5, -1., 5, 4),
        (5, 1., 6, 1), (5, -1., 7, 0), (5, -1., 7, 8), (5, 1., 8, 7)]
}


def fc_fce(mode, coors, val=1):
    if not (mode == 'qp'):
        return

    tstep = hyperela['step']
    print('### loading function - step %f' % tstep)

    nc, dim = coors.shape
    out = nm.zeros((nc, dim, 1), dtype=nm.float64)
    out[:, dim - 1, 0] = -val

    if (tstep % 1) > 0:
        out *= 0

    return {'val': out}


def initial_problem(pb):
    from sfepy.base.conf import ProblemConf, get_standard_keywords
    from sfepy.discrete import Problem

    out_pref = output.get_output_prefix()
    output.set_output_prefix(out_pref + ' <init pb>:')

    required, other = get_standard_keywords()

    cdict = pb.conf.funmod.define()

    par_s = cdict['material_params']['solid']
    par_f = cdict['material_params']['fluid']
    par = cdict['material_params']['param']
    dim = cdict['params']['dim']

    mat_Phi = 1 - par_s['Phi']
    mat_rho = mat_Phi * par_f['rho'] + par_s['Phi'] * par_s['rho']
    mat_K = par_f['kappa'] / (par['g'] * par_f['rho']) * nm.eye(dim)
    mat_M = mat_Phi / par_f['Kf']

    cdict.update({
        'options': {},
        'materials': {
            'mat': ({
                'rho': mat_rho,
                'rhofK': par_f['rho'] * mat_K,
                'M': mat_M,
            },),
            'force': ({'val': -cdict['params']['force']},),
        },
        'variables': {
            'u': ('unknown field', 'displacement'),
            'v': ('test field', 'displacement', 'u'),
            'p': ('unknown field', 'pressure'),
            'q': ('test field', 'pressure', 'p'),
        },
        'equations': {
            'eq1': """dw_volume_dot.5.Omega(mat.rho, v, u)
                    = dw_surface_ltr.5.ForceRegion(force.val, v)""",
            'eq2': """dw_volume_dot.5.Omega(mat.M, q, p)
                    + dw_v_dot_grad_s.5.Omega(mat.rhofK, u, q) = 0"""
        },
    })

    conf = ProblemConf.from_dict(cdict, pb.conf.funmod, required, other)
    init_pb = Problem.from_conf(conf)
    init_pb.time_update()
    res = init_pb.solve().get_parts()

    output.set_output_prefix(out_pref)
    return res['u'], res['p']


def post_process(time, vals, lab=''):
    import matplotlib.pyplot as plt

    time0 = nm.array([0] + list(time))
    for k, v in vals.items():
        if len(v) > 1:
            ndid, val = int(v[0]), nm.asarray(v[1:])
            fig = plt.figure()
            plt.ylabel('$%s_{nd%d}$' % (k, ndid), fontsize=16)
            plt.xlabel('$t [s]$', fontsize=16)
            plt.plot(time0, val)
            plt.grid(True)
            fig.savefig('pp_%s_%s.png' % (lab, k))


def incremental_algorithm(pb):
    params = pb.conf.params
    dt, t_end = params['dt'], params['t_end']
    st_nd_u = params.get('u_store_node')
    st_nd_p = params.get('p_store_node')

    hist = {'p': [], 'u': []}
    hist['u'].append(st_nd_u)
    hist['p'].append(st_nd_p)

    timehist = nm.arange(0, t_end, dt) + dt
    pbvars = pb.get_variables()
    pb.domain.mesh.coors_act = pb.domain.mesh.coors.copy()

    state = hyperela['state']

    out = []

    aux = pbvars['u'].field.get_coor()
    dim = aux.shape[1]

    if params['init_mode']:
        du2, dp = initial_problem(pb)
    else:
        du2, dp = nm.zeros_like(aux), 0

    state['coors0'] = aux.copy()
    state['u'] = nm.zeros_like(aux)
    state['du'] = nm.zeros_like(aux)
    state['du1'] = nm.zeros_like(aux)
    aux = pbvars['p'].field.get_coor()[:, 0].squeeze()
    state['p'] = nm.zeros_like(aux)
    state['dp'] = nm.zeros_like(aux)
    nnod = aux.shape[0]

    if st_nd_u is not None:
        hist['u'].append(state['u'][st_nd_u, dim - 1])
    if st_nd_p is not None:
        hist['p'].append(state['p'][st_nd_p])

    state['du1'][:] = state['du'] - du2.reshape(state['du'].shape) * dt**2
    state['dp'][:] = dp * dt

    for it0, time in enumerate(timehist):
        it = it0 + 1
        print('>>> step %d:' % it)
        hyperela['step'] = it

        pbvars['P'].set_data(state['p'][:])
        pbvars['dU'].set_data(state['du'][:])

        yield pb, out

        result = out[-1][1].get_parts()
        du = result['u']
        dp = result['p']

        state['u'] += du.reshape(state['du'].shape)
        state['p'] += dp

        state['du'][:] = du.reshape(state['du'].shape)
        state['dp'][:] = dp

        pb.set_mesh_coors(state['u'] + state['coors0'],
                          update_fields=True, actual=True)

        if params['save_step']:
            pbvars['P'].set_data(state['p'][:])
            w = pb.evaluate('ev_diffusion_velocity.5.Omega(mat.K, P)',
                            mode='el_avg')
            J = hyperela['aux']['J']
            tout = {}
            tout['u'] = Struct(name='output_data',
                               mode='vertex',
                               data=state['u'][:nnod, :],
                               dofs=None)
            tout['p'] = Struct(name='output_data',
                               mode='vertex',
                               data=state['p'].reshape((nnod, 1)),
                               dofs=None)
            tout['w'] = Struct(name='output_data',
                               mode='cell',
                               data=w,
                               dofs=None)
            tout['J'] = Struct(name='output_data',
                               mode='cell',
                               data=J.reshape((J.shape[0], 1, 1, 1)),
                               dofs=None)

            out_fn = osp.join(pb.output_dir, '%s_%03d.vtk'
                              % (osp.split(pb.domain.mesh.name)[1], it))
            pb.domain.mesh.write(out_fn, out=tout)

        if st_nd_u is not None:
            hist['u'].append(state['u'][st_nd_u, dim - 1])
        if st_nd_p is not None:
            hist['p'].append(state['p'][st_nd_p])

        pbvars['P'].set_data(state['p'][:])

        yield None
        print('<<< step %d finished' % it)

    # postprocess
    post_process(timehist, hist, 'he')


def append_mat_hist(qp, hdict):
    mhist = hyperela['mhist']
    for k, v in hdict.items():
        if k not in mhist:
            mhist[k] = []
        mhist[k].append(v[qp])


def mat_fce(mode, coors, term, pb):
    if not (mode == 'qp'):
        return

    print('### material function - step %f' % hyperela['step'])

    cache = hyperela['cache']
    # clear chache
    if hyperela['step'] != hyperela['step_in_cache']:
        cache.clear()

    cache_key = (hyperela['step'], term.region.name,
                 term.integral.name, term.integration)

    if cache_key not in cache:
        npts = coors.shape[0]

        state = pb.create_variables(['Um', 'Pm'])
        state_u = state['Um']
        state_p = state['Pm']

        n_el, n_qp, dim, n_en, n_c = \
            state_u.get_data_shape(term.integral, term.integration,
                                   term.region.name)
        dim2 = dim**2

        state_u.set_data(hyperela['state']['u'])
        state_p.set_data(hyperela['state']['p'])

        fd = HyperElasticULFamilyData()

        if hyperela['mapping0'] is None:
            family_data = fd(state_u,
                             term.region, term.integral, term.integration)
            hyperela['mapping0'] = state_u.field.mappings.copy()
        else:
            state_u.field.mappings0 = hyperela['mapping0']
            family_data = fd(state_u,
                             term.region, term.integral, term.integration)
            state_u.field.mappings0 = {}

        J = family_data.det_f
        b = family_data.sym_b

        # du - displacement increment
        state_u.set_data(hyperela['state']['du'])
        # grad(du)^T
        grad_duT = state_u.evaluate(mode='grad',
                                    region=term.region,
                                    integral=term.integral,
                                    integration=term.integration)
        # grad(du)
        grad_du = grad_duT.swapaxes(2, 3)
        # div(du)
        div_du = state_u.evaluate(mode='div',
                                  region=term.region,
                                  integral=term.integral,
                                  integration=term.integration)
        # p - pressure in QP
        state_p.set_data(hyperela['state']['p'])
        p_qp = state_p.evaluate(mode='val',
                                region=term.region,
                                integral=term.integral,
                                integration=term.integration)
        grad_p = state_p.evaluate(mode='grad',
                                  region=term.region,
                                  integral=term.integral,
                                  integration=term.integration)
        # dp - pressure increment in QP
        state_p.set_data(hyperela['state']['dp'])
        dp_qp = state_p.evaluate(mode='val',
                                 region=term.region,
                                 integral=term.integral,
                                 integration=term.integration)

        par_s = pb.conf.material_params['solid']
        par_f = pb.conf.material_params['fluid']
        par = pb.conf.material_params['param']

        mat_rho_f = par_f['rho'] * nm.exp(p_qp / par_f['Kf'])
        mat_Phi_s = par_s['Phi'] / J
        mat_Phi = 1 - mat_Phi_s
        mat_rho = par_s['rho'] * mat_Phi_s + mat_rho_f * mat_Phi

        # B
        mat_B = nm.tile(par['B'], (n_el, n_qp, 1, 1))
        mat_Bsym = mat_B.reshape((n_el, n_qp, dim2, 1))[:, :, ssym_tab[dim], :]
        # mat_tB = dot_sequences(mat_B, (div_du - grad_duT))  # ?? ^T
        mat_tB = div_du * mat_B - dot_sequences(mat_B, grad_duT)  # ?? ^T, 29.6.2020

        # K
        par_K = par_f['kappa'] / (par['g'] * par_f['rho'])
        mat_K = par_K * nm.exp(par_f['beta'] * (J - 1)) * nm.eye(dim)
        mat_dK = par_f['beta'] * J * div_du * mat_K
        Kdu = dot_sequences(mat_K, grad_duT)
        mat_tK = mat_K * div_du - Kdu - Kdu.swapaxes(2, 3)
        mat_rhofK = mat_rho_f * mat_K
        duK = dot_sequences(grad_du, mat_K)
        mat_rhofKupdK = mat_rho_f * (mat_K * (1 + dp_qp/par_f['Kf'] + div_du)
                                     - duK + mat_dK)

        # M
        mat_M = mat_Phi / par_f['Kf']
        mat_MM = mat_M * (1 + div_du)
        mat_MMM = mat_MM - par_s['rho'] / (par_f['Kf'] * J) * div_du

        mat_R = mat_rho_f * (div_du + mat_Phi / par_f['Kf'] * dp_qp)

        # A
        mat_mu = nm.ones((n_el, n_qp, 1, 1)) * par_s['mu']
        mat_lam = nm.ones((n_el, n_qp, 1, 1)) * par_s['lam']

        mat_mux = mat_mu - mat_lam * nm.log(J)
        tanmod = mat_mux * delta_delta[dim] + mat_lam * one_one[dim]
        stress = mat_mu * b + (mat_lam * nm.log(J) - mat_mu) * sym_eye[dim]

        # \tau^eff_il * \delta_jk
        stress_eff_d2d2 = nm.zeros((n_el, n_qp, dim2, dim2), dtype=nm.float64)
        for ii, jj, kk in stress_Atab[dim]:
            stress_eff_d2d2[:, :, jj, kk] = stress[:, :, ii, 0]
        # \hat{p} * (B_il * \delta_ik - B_ij * \delta_kl)
        press_d2d2 = nm.zeros((n_el, n_qp, dim2, dim2), dtype=nm.float64)
        for ii, sg, jj, kk in press_Atab[dim]:
            press_d2d2[:, :, jj, kk] =\
                p_qp[:, :, 0, 0] * mat_Bsym[:, :, ii, 0] * sg

        mat_A = (tanmod + stress_eff_d2d2) / J + press_d2d2

        stress -= p_qp * mat_Bsym * J

        cache[cache_key] = {
            'A': mat_A.reshape((npts, dim2, dim2)),
            'S': (stress / J).reshape(npts, stress.shape[2], 1),
            'B': mat_B.reshape(npts, dim, dim),
            'BB': (mat_B + mat_tB).reshape(npts, dim, dim),
            'K': mat_K.reshape(npts, dim, dim),
            'KKK': (mat_K + mat_tK + mat_dK).reshape(npts, dim, dim),
            'M': mat_M.reshape(npts, 1, 1),
            'MMM': mat_MMM.reshape(npts, 1, 1),
            'rho': mat_rho.reshape(npts, 1, 1),
            'rhoR': (mat_rho + mat_R).reshape(npts, 1, 1),
            'rhofK': mat_rhofK.reshape(npts, dim, dim),
            'rhofKupdK': mat_rhofKupdK.reshape(npts, dim, dim),
        }

        tr_b = nm.trace(b, axis1=2, axis2=3).reshape((n_el, n_qp, 1, 1))
        psi = mat_mu * ((tr_b - 3) / 2 - nm.log(J))\
            + mat_lam / 2 * nm.log(J)**2
        w = - dot_sequences(mat_K, grad_p)

        hyperela['aux']['J'] = nm.average(J, axis=1)

        hyperela['step_in_cache'] = hyperela['step']
        print('  computed')

        elem = pb.conf.params['mat_store_elem']
        append_mat_hist((elem, 0, 0, 0), {
            'rho': mat_rho,
            'R': mat_R,
            'K': mat_K,
            'J': J,
            'w': w,
            'psi': psi,
            'M': mat_M,
            'p': p_qp,
            })
        append_mat_hist((elem, 0, 0, 0), {'S_11': stress})
        append_mat_hist((elem, 0, 1, 0), {'S_22': stress})
        if dim > 2:
            append_mat_hist((elem, 0, 2, 0), {'S_33': stress})
    else:
        print('  cached')

    return copy.deepcopy(cache[cache_key])


def def_problem(dt, fc_val=None, displ_val=None):
    materials = {
        'mat': 'mat_fce',
    }

    functions = {
        'mat_fce': (lambda ts, coors, mode=None, term=None,
            problem=None, **kwargs: mat_fce(mode, coors, term, problem),),
    }

    if fc_val is not None:
        materials['force'] = 'fc_fce'
        functions['fc_fce'] = (lambda ts, coors, mode=None, **kwargs:
                               fc_fce(mode, coors, fc_val),)
    if displ_val is not None:
        materials['displ'] = 'displ_fce'
        functions['displ_fce'] = (lambda ts, coors, mode=None, **kwargs:
                                  fc_fce(mode, coors, displ_val),)

    fields = {
        'displacement': ('real', 'vector', 'Omega', 2),
        'pressure': ('real', 'scalar', 'Omega', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'U': ('parameter field', 'displacement', '(set-to-None)'),
        'dU': ('parameter field', 'displacement', '(set-to-None)'),
        'Um': ('parameter field', 'displacement', '(set-to-None)'),
        'P': ('parameter field', 'pressure', '(set-to-None)'),
        'Pm': ('parameter field', 'pressure', '(set-to-None)'),
    }

    dti = dt**-1

    fc_term = 'dw_surface_ltr.5.ForceRegion(force.val, v)'\
              if fc_val is not None else ''
    equations = {
        'balance_of_forces':
            """
          %e * dw_volume_dot.5.Omega(mat.rhoR, v, u)
             + dw_nonsym_elastic.5.Omega(mat.A, v, u)
             - dw_biot.5.Omega(mat.B, v, p)
             = %s
             - dw_lin_prestress.5.Omega(mat.S, v)
        + %e * dw_volume_dot.5.Omega(mat.rhoR, v, dU)"""\
            % (dti**2, fc_term, dti**2),
        'fluid':
            """
          %e * dw_v_dot_grad_s.5.Omega(mat.rhofKupdK, u, q)
             + dw_volume_dot.5.Omega(mat.MMM, q, p)
             + dw_biot.5.Omega(mat.BB, u, q)
        + %e * dw_diffusion.5.Omega(mat.K, q, p)
             =
          %e * dw_v_dot_grad_s.5.Omega(mat.rhofKupdK, dU, q)
        - %e * dw_diffusion.5.Omega(mat.KKK, q, P)""" % (dti, dt, dti, dt),
    }

    solvers = {
        'ls': ('ls.scipy_direct', {}),
        'newton': ('nls.newton', {'eps_a': 1e-3, 'eps_r': 1e-3}),
    }

    return materials, functions, fields, variables, equations, solvers
