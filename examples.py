import numpy as onp
import jax
import jax.numpy as np
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from jax.lib import xla_bridge
import time
from jax_dem.optimizer import optimize, simulate, leapfrog
from jax_dem.arguments import args
from jax_dem.dynamics import get_state_rhs_func, drum_env, box_env, drum_env, ptcl_state_rhs_func_prm, \
obj_state_rhs_func_prm_nonuniform, obj_state_rhs_func_prm_uniform, obj_to_ptcl_uniform_batch, test_energy, obj_states_to_ptcl_states
from jax_dem.io import vedo_plot, plot_energy
from jax_dem.utils import quat_mul
import matplotlib.pyplot as plt

dim = args.dim

def walltime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print(f"Time elapsed {end_time-start_time} on platform {xla_bridge.get_backend().platform}") 
    return wrapper


@walltime
def single_bouncing_ball():
    case_name = 'single_bouncing_ball'
    n_objects = 1
    radius = 0.5 

    nondiff_kwargs = {'gravity': 0.,
                      'radii': radius * np.ones(n_objects),
                      'normal_contact_stiffness': 1e4,
                      'damping_coeff': 0.,
                      'Coulomb_fric_coeff': 0.,
                      'tangent_fric_coeff': 0.,
                      'rolling_fric_coeff': 0.,
                      'box_env_bottom': 10., 
                      'box_env_top': 90.}

    # ts = np.arange(0., 1, 1e-2)
    ts = np.arange(0., 0.1, 0.2*1e-3)
    x0 = np.array([50, 50, nondiff_kwargs['box_env_bottom'] + 10*radius]).reshape(-1, 1)
    v0 = np.array([0, 0, -(9 + 4)*radius/0.1]).reshape(-1, 1)
    q0 = np.ones((1, n_objects))
    y0 = np.concatenate([x0, q0, np.zeros((3, n_objects)), v0, np.zeros((3, n_objects))], axis=0).T

    def obj_func(yf):
        x = yf[:, 0:3]
        return x[0, 2]

    diff_kwargs = {} 
    diff_keys = tuple(diff_kwargs.keys())
    diff_args = tuple(diff_kwargs.values())
    state_rhs_func = get_state_rhs_func(ptcl_state_rhs_func_prm, diff_keys, box_env, nondiff_kwargs)

    bounds = onp.vstack([y0, y0]).T
    bounds[2, :] = np.array([radius, nondiff_kwargs['box_env_bottom'] + 30*radius])

    # ys = simulate([y0, diff_args], ts, state_rhs_func)
    # vedo_plot(case_name, nondiff_kwargs['radii'], nondiff_kwargs['box_env_bottom'], nondiff_kwargs['box_env_top'], ys)

    optimize([y0, diff_args], ts, obj_func, state_rhs_func, bounds)


@walltime
def billiards():
    case_name = 'billiards'
    n_objects = 11
    radius = 0.5 
    nondiff_kwargs = {'radii': radius * np.ones(n_objects),
                      # 'normal_contact_stiffness': 1e4,
                      # 'damping_coeff': 5,
                      # 'Coulomb_fric_coeff': 0.5,
                      # 'tangent_fric_coeff': 1e1,
                      # 'rolling_fric_coeff': 0.05,

                      'normal_contact_stiffness': 1e4,
                      'damping_coeff': 0.,
                      'Coulomb_fric_coeff': 0.,
                      'tangent_fric_coeff': 0.,
                      'rolling_fric_coeff': 0.,

                      'box_env_bottom': 10., 
                      'box_env_top': 90., }

    dt = 1e-3
    ts = np.arange(0., dt*301, dt)

    intersect = 4./3.*np.pi*radius**3 / nondiff_kwargs['normal_contact_stiffness']
    base_x = 50.
    base_y = 50.
    a = 1.05*radius
    b = np.sqrt(3.) * a
    x0_z = radius - intersect + nondiff_kwargs['box_env_bottom']
    # x0 = np.array([[base_x, base_x, base_x-a, base_x+a, base_x-2*a, base_x,     base_x+2*a, base_x-3*a, base_x-a,   base_x+a,   base_x+3*a],
    #                [48.,    base_y, base_y+b, base_y+b, base_y+2*b, base_y+2*b, base_y+2*b, base_y+3*b, base_y+3*b, base_y+3*b, base_y+3*b],
    #                [x0_z,   x0_z,   x0_z,     x0_z,     x0_z,       x0_z,       x0_z,       x0_z,       x0_z,       x0_z,       x0_z      ]])
    billiard_layers = 4
    x0 = np.array([[base_x - i*a + 2*a*j, base_y + i*b, x0_z] for i in range(billiard_layers) for j in range(i + 1)])
    x0 = np.vstack((np.array([[base_x, 48., x0_z]]), x0)).T

    v0 = np.hstack((np.array([[0., 10., 0.]]).T, np.zeros((dim, n_objects - 1))))
    q0 = np.ones((1, n_objects))
    y0 = np.concatenate([x0, q0, np.zeros((3, n_objects)), v0, np.zeros((3, n_objects))], axis=0).T

    def obj_func(yf):
        xf = yf[-1, 0:2]
        target_xf = np.array([base_x + 3*a + 5*a, base_y + 3*b + 5*b])
        # target_xf = np.array([53., 55.])
        print(f"target_xf = {target_xf}, current_xf = {xf}, l2 loss = {np.sum((xf - target_xf)**2)}")
        return np.sum((xf - target_xf)**2)

    diff_kwargs = {} 
    diff_keys = tuple(diff_kwargs.keys())
    diff_args = tuple(diff_kwargs.values())
    state_rhs_func = get_state_rhs_func(ptcl_state_rhs_func_prm, diff_keys, box_env, nondiff_kwargs)

    def fwd_sim():

        def finite_difference():
            ys = simulate([y0, diff_args], ts, state_rhs_func)
            obj1 = obj_func(ys[-1])
            hs = [0.1 * 0.5**i for i in range(12)]
            for h in hs:
                # y0 = jax.ops.index_update(y0, jax.ops.index[(7, 8), (0, 0)], np.array([0, 10 + h]))
                y0 = jax.ops.index_update(y0, jax.ops.index[(0,), (1,)], 48. + h)
                ys = simulate([y0, diff_args], ts, state_rhs_func)
                obj2 = obj_func(ys[-1])
                fd_der = (obj2 - obj1) / h
                print(f"finite difference derivative = {fd_der} with h = {h}")

        def bwd_solve():
            ys = simulate([y0, diff_args], ts, state_rhs_func)
            ys_bwd = simulate([ys[-1], diff_args], -ts[::-1], lambda state, t, *diff_args: -state_rhs_func(state, t, *diff_args))
            print(y0)
            print("\n")
            print(ys[-1])
            print("\n")
            print(ys_bwd[-1])
            fig = plt.figure()
            plt.plot(ts[-2::-1], ys_bwd[:, 0, 10], linestyle='-', marker='o', color='red')
            plt.plot(ts[1:], ys[:, 0, 10], linestyle='-', marker='o', color='blue')
            plt.show()

        def simple_solve():
            ys = simulate([y0, diff_args], ts, state_rhs_func)
            np.save(f'data/numpy/vedo/states_{case_name}.npy', ys)
            vedo_plot(case_name, nondiff_kwargs['radii'], nondiff_kwargs['box_env_bottom'], nondiff_kwargs['box_env_top'], ys[::20]) 

        simple_solve()

    def inv_opt():
        def diff_through_loops():
            def sim(y0):
                ys = simulate([y0, diff_args], ts, state_rhs_func)
                return obj_func(ys[-1])
            grad_sim = jax.grad(sim)
            return grad_sim

        grads = diff_through_loops()(y0)
        print(grads)

        lower_bound, _ = ravel_pytree([jax.ops.index_update(y0, jax.ops.index[(0, 0), (7, 8)], np.array([-50, -50])), diff_args])
        upper_bound, _ = ravel_pytree([jax.ops.index_update(y0, jax.ops.index[(0, 0), (7, 8)], np.array([50., 50.])), diff_args])
        # lower_bound, _ = ravel_pytree([jax.ops.index_update(y0, jax.ops.index[(0, 0, 0 ,0), (0, 1, 7, 8)], np.array([25, 25, -50, -50])), diff_args])
        # upper_bound, _ = ravel_pytree([jax.ops.index_update(y0, jax.ops.index[(0, 0, 0, 0), (0, 1, 7, 8)], np.array([75, 75, 50., 50.])), diff_args])
        bounds = onp.vstack([lower_bound, upper_bound]).T
        y0_opt, diff_args_opt = optimize([y0, diff_args], ts, obj_func, state_rhs_func, bounds)

        ys = simulate([y0_opt, diff_args_opt], ts, state_rhs_func)
        vedo_plot(case_name, nondiff_kwargs['radii'], nondiff_kwargs['box_env_bottom'], nondiff_kwargs['box_env_top'], ys[::20]) 

    # fwd_sim()
    inv_opt()


@walltime
def particles_in_box():
    case_name = 'particles_in_box'
    nondiff_kwargs = {'box_env_bottom': 10., 'box_env_top': 90.}
    env_bottom = nondiff_kwargs['box_env_bottom']
    env_top = nondiff_kwargs['box_env_top']

    # dt = 1e-3
    dt = 2*1e-3
    ts = np.arange(0., dt*100, dt)
    radius = 0.5

    n_objects_axis = 20
    spacing = np.linspace(env_bottom + 0.1*(env_top - env_bottom), env_top - 0.1*(env_top - env_bottom), n_objects_axis)
    # spacing = np.linspace(env_bottom + 2*radius, env_bottom + (4*n_objects_axis - 2)*radius, n_objects_axis)

    n_objects = len(spacing)**3
    x1, x2, x3 = np.meshgrid(*([spacing]*3), indexing='ij')
    key = jax.random.PRNGKey(0)
    perturb = jax.random.uniform(key, (dim, n_objects), np.float32, -0.5*radius, 0.5*radius)
    x0 = np.concatenate([x1.reshape(1, -1), x2.reshape(1, -1), x3.reshape(1, -1)], axis=0) + perturb
    q0 = np.ones((1, n_objects))
    y0 = np.concatenate([x0, q0, np.zeros((9, n_objects))], axis=0).T
    radii = radius * np.ones(n_objects)

    diff_kwargs = {'radii': radii}
    diff_keys = tuple(diff_kwargs.keys())
    diff_args = tuple(diff_kwargs.values())
    state_rhs_func = get_state_rhs_func(ptcl_state_rhs_func_prm, diff_keys, box_env, nondiff_kwargs)
    ys = simulate([y0, diff_args], ts, state_rhs_func)
    
    # np.save(f'data/numpy/vedo/states_{case_name}.npy', ys)
    # vedo_plot(case_name, diff_kwargs['radii'], env_bottom, env_top, states=None)


@walltime
def particles_in_drum():
    case_name = 'particles_in_drum'
    radius = 0.5
    n_objects_axis = 10
    # spacing = np.linspace(44.5, 55, n_objects_axis)
    spacing = np.linspace(45, 55, n_objects_axis)
    n_objects = len(spacing)**3    
    nondiff_kwargs = {'radii': radius * np.ones(n_objects), 
                      'normal_contact_stiffness': 1e4,
                      'damping_coeff': 1e1,
                      'Coulomb_fric_coeff': 0.5,
                      'tangent_fric_coeff': 1e1,
                      'rolling_fric_coeff': 0.2,
                      'box_env_bottom': 10., 
                      'box_env_top': 90.}
    env_bottom = nondiff_kwargs['box_env_bottom']
    env_top = nondiff_kwargs['box_env_top']


    x1, x2, x3 = np.meshgrid(*([spacing]*3), indexing='ij')
    x0 = np.concatenate([x1.reshape(1, -1), x2.reshape(1, -1), x3.reshape(1, -1)], axis=0)
    q0 = np.ones((1, n_objects))
    y0 = np.concatenate([x0, q0, np.zeros((9, n_objects))], axis=0).T
 
    def obj_func(yf):
        xf = yf[:, :3]
        center = np.mean(xf, axis=0).reshape(1, dim)
        return -np.sum((xf - center)**2)

    def fwd_sim():
        # dt = 1e-4
        dt = 2*1e-3
        ts = np.arange(0., dt*4001, dt)
        # ts = np.linspace(0., 8., 2001)
        # fig = plt.figure()
        # omegas = np.array([4./25.*np.pi])
        omegas = np.linspace(2./25.*np.pi, 26/25.*np.pi, 7)
        objs = []
        for i, omega in enumerate(omegas):
            print(f"Meta step {i}， omega = {omega}")
            diff_kwargs = {'drum_env_omega': np.array([omega, 0., 0.])}
            state_rhs_func = get_state_rhs_func(ptcl_state_rhs_func_prm, tuple(diff_kwargs.keys()), drum_env, nondiff_kwargs)
            ys = simulate([y0, tuple(diff_kwargs.values())], ts, state_rhs_func)
            # vedo_plot(case_name, nondiff_kwargs['radii'], env_bottom, env_top, ys[::20])

            plt.plot(ts[1::20], -jax.vmap(obj_func)(ys[::20]), linestyle='-', marker='o', label=str(i))
            objs.append(-jax.vmap(obj_func)(ys))

        objs = np.stack(objs)
        print(objs.shape)
        print(objs)
        np.save(f'data/numpy/vedo/objectives_{case_name}.npy', objs)
        plt.legend(fontsize=16, frameon=False, loc='upper right')
        plt.show()

    def inv_opt():
        dt = 2*1e-3
        ts = np.arange(0., dt*4001, dt)
        # ts = np.linspace(0., 8., 2001)
        diff_kwargs = {'drum_env_omega': np.array([16./25.*np.pi, 0., 0.])}
        state_rhs_func = get_state_rhs_func(ptcl_state_rhs_func_prm, tuple(diff_kwargs.keys()), drum_env, nondiff_kwargs)
        lower_bound, _ = ravel_pytree([y0, np.array([0, 0., 0.])])
        upper_bound, _ = ravel_pytree([y0, np.array([100./25.*np.pi, 0., 0.])])
        bounds = onp.vstack([lower_bound, upper_bound]).T
        optimize([y0, tuple(diff_kwargs.values())], ts, obj_func, state_rhs_func, bounds)


    def experiment():
        dt = 2*1e-3
        ts = np.arange(0., dt*10001, dt)
        diff_kwargs = {'drum_env_omega': np.array([6./25.*np.pi, 0., 0.])}
        diff_args = tuple(diff_kwargs.values())
        state_rhs_func = get_state_rhs_func(ptcl_state_rhs_func_prm, tuple(diff_kwargs.keys()), drum_env, nondiff_kwargs)

        # ys = simulate([y0, diff_args], ts, state_rhs_func)
        # np.save(f'data/numpy/vedo/states_{case_name}.npy', ys)
        ys = np.load(f'data/numpy/vedo/states_{case_name}.npy')

        rev_ts = ts[::-1]
        rev_ys = np.vstack((y0[None, ...], ys))[::-1]


        y_bar, diff_args_bar = jax.grad(obj_func)(rev_ys[0]), tree_map(np.zeros_like, diff_args)
        y_bars = [y_bar]
        diff_args_bars = [diff_args_bar]
        # diff_args_bars_decoupled = []

        for i in range(201):

            y_prev = rev_ys[i + 1]
            t_prev = rev_ts[i + 1]
            y_crt = rev_ys[i]
            t_crt = rev_ts[i]

            y_dot, vjpfun = jax.vjp(lambda y_prev, *diff_args: leapfrog((y_prev, t_prev), t_crt, state_rhs_func, *diff_args)[1], y_prev, *diff_args)
            y_bar, *diff_args_bar = vjpfun(y_bar)

            # _, *diff_args_bar_decoupled = vjpfun(jax.grad(obj_func)(y_crt))

            y_bars.append(y_bar)
            diff_args_bars.append(tuple(diff_args_bar))
            # diff_args_bars_decoupled.append(tuple(diff_args_bar_decoupled))

            if i % 20 == 0:
                print(f"Reverse step {i}") 
                if not np.all(np.isfinite(y_bar)):
                    print(f"Found np.inf or np.nan in y - stop the program")             
                    # exit()

        y_bars = np.stack(y_bars)
        diff_args_bars = jax.tree_multimap(lambda *xs: np.stack(xs), *diff_args_bars)[0]
        # diff_args_bars_decoupled = jax.tree_multimap(lambda *xs: np.stack(xs), *diff_args_bars_decoupled)[0]

        cumsum_diff_args_bars = np.cumsum(diff_args_bars, axis=0)
        print(f"diff_args_bars = \n{diff_args_bars}")
        print(f"cumsum = \n{cumsum_diff_args_bars}")

        # fig = plt.figure()
        # plt.plot(np.arange(len(diff_args_bars_decoupled)), diff_args_bars_decoupled[:, 0], linestyle='-', marker='o', markersize=2)

        fig = plt.figure()
        plt.plot(np.arange(len(cumsum_diff_args_bars)), cumsum_diff_args_bars[:, 0], linestyle='-', marker='o', markersize=2)
        plt.show()

    # fwd_sim()
    # inv_opt()
    experiment()


@walltime
def objects_in_drum():
    case_name = 'objects_in_drum'

    radius = 0.5
    n_objects_axis = 5
    spacing = np.linspace(46, 54.5, n_objects_axis)
    n_objects = len(spacing)**3    

    x0 = np.stack(np.meshgrid(*([spacing]*3), indexing='ij')).reshape(dim, -1).T
    q0 = np.ones((n_objects, 1))
    y0 = np.hstack((x0, q0, np.zeros((n_objects, 9))))

    ptcl_arm_ref_each = np.stack(np.meshgrid(*([np.array([-radius, radius])]*3), indexing='ij')).reshape(dim, -1).T
    ptcl_arm_ref = list(np.repeat(ptcl_arm_ref_each[None, ...], n_objects, axis=0))
    radii_list = jax.tree_map(lambda x: radius*np.ones((x.shape[0])), ptcl_arm_ref)
    ptcl_split = np.cumsum(np.array(jax.tree_map(lambda x: x.shape[0], radii_list)))[:-1]

    nondiff_kwargs = {'normal_contact_stiffness': 1e4,
                      'damping_coeff': 1e1,
                      'Coulomb_fric_coeff': 0.5,
                      'tangent_fric_coeff': 1e1,
                      'rolling_fric_coeff': 0.2,
                      'box_env_bottom': 10., 
                      'box_env_top': 90.,
                      # 'ptcl_arm_ref': ptcl_arm_ref,
                      # 'radii_list': radii_list,
                      # 'ptcl_split': ptcl_split,
                      'ptcl_arm_ref_arr': np.stack(ptcl_arm_ref),
                      'radii_arr': np.stack(radii_list)}
    env_bottom = nondiff_kwargs['box_env_bottom']
    env_top = nondiff_kwargs['box_env_top']


    def fwd_sim():
        dt = 2*1e-3
        ts = np.arange(0., dt*4001, dt)
        diff_kwargs = {'drum_env_omega': np.array([15./25.*np.pi, 0., 0.])}
        state_rhs_func = get_state_rhs_func(obj_state_rhs_func_prm_uniform, tuple(diff_kwargs.keys()), drum_env, nondiff_kwargs)

        obj_ys = simulate([y0, tuple(diff_kwargs.values())], ts, state_rhs_func)
        np.save(f'data/numpy/vedo/states_{case_name}.npy', obj_ys[::20])
        obj_ys = np.load(f'data/numpy/vedo/states_{case_name}.npy')
        ptcl_ys = obj_states_to_ptcl_states(obj_ys, nondiff_kwargs['ptcl_arm_ref_arr'])
        vedo_plot(case_name, radius, env_bottom, env_top, ptcl_ys[::20])

    fwd_sim()


@walltime
def donuts():
    case_name = 'donuts'

    radius = 0.5

    # n_objects_axis = 10
    n_objects_axis = 2

    n_ptcl_per_obj = 20
    donut_radius = n_ptcl_per_obj * 4./3. * np.pi * radius**3 / (np.pi * radius**2 * 2 * np.pi)
    outer_diamter = 2 * (donut_radius + radius)

    # spacing = np.linspace(50 - 5 * outer_diamter, 50 + 5 * outer_diamter, n_objects_axis)
    # spacing = np.linspace(25 - 2.1 * outer_diamter, 25 + 2.1 * outer_diamter, n_objects_axis)
    spacing = np.linspace(22 - 1 * outer_diamter, 22 + 1 * outer_diamter, n_objects_axis)


    n_objects = len(spacing)**3 
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / n_ptcl_per_obj)
    ptcl_arm_ref_each = np.vstack((donut_radius * np.cos(angles), donut_radius * np.sin(angles), np.zeros(n_ptcl_per_obj))).T
    ptcl_arm_ref = list(np.repeat(ptcl_arm_ref_each[None, ...], n_objects, axis=0))
    radii_list = jax.tree_map(lambda x: radius*np.ones((x.shape[0])), ptcl_arm_ref)
 
    x0 = np.stack(np.meshgrid(*([spacing]*3), indexing='ij')).reshape(dim, -1).T
    key = jax.random.PRNGKey(0)
    q0 = jax.random.normal(key, (n_objects, dim + 1))
    q0 /= np.linalg.norm(q0, axis=1)[:, None]
    y0 = np.hstack((x0, q0, np.zeros((n_objects, 6))))

    nondiff_kwargs = {'normal_contact_stiffness': 1e4,
                      # 'damping_coeff': 1e1,
                      # 'Coulomb_fric_coeff': 0.5,
                      # 'tangent_fric_coeff': 1e1,
                      # 'rolling_fric_coeff': 0.2,

                      'damping_coeff': 0.,
                      'Coulomb_fric_coeff': 0.,
                      'tangent_fric_coeff': 0.,
                      'rolling_fric_coeff': 0.,

                      'box_env_bottom': 10., 
                      'box_env_top': 90.,
                      'ptcl_arm_ref_arr': np.stack(ptcl_arm_ref),
                      'radii_arr': np.stack(radii_list)}

    env_bottom = nondiff_kwargs['box_env_bottom']
    env_top = nondiff_kwargs['box_env_top']


    def fwd_sim():
        dt = 2 * 1e-3
        ts = np.arange(0., dt*1001, dt)
        diff_kwargs = {'drum_env_omega': np.array([15./25.*np.pi, 0., 0.])}
        state_rhs_func = get_state_rhs_func(obj_state_rhs_func_prm_uniform, tuple(diff_kwargs.keys()), box_env, nondiff_kwargs)

        obj_ys = simulate([y0, tuple(diff_kwargs.values())], ts, state_rhs_func)
        obj_ys_skip = obj_ys[::20]
        np.save(f'data/numpy/vedo/states_{case_name}.npy', obj_ys_skip)

        # obj_ys_skip = np.load(f'data/numpy/vedo/states_{case_name}.npy')

        ptcl_ys_skip = obj_states_to_ptcl_states(obj_ys_skip, nondiff_kwargs['ptcl_arm_ref_arr'])

        energy = test_energy(ptcl_ys_skip, nondiff_kwargs['radii_arr'].reshape(-1))
        plot_energy(energy, f'data/pdf/energy_{case_name}.pdf')
        vedo_plot(case_name, radius, env_bottom, env_top, ptcl_ys_skip)

    fwd_sim()



@walltime
def donut_with_rope():
    case_name = 'donut_with_rope'

    radius = 0.5

    # n_objects_axis = 10
    n_objects_axis = 2

    n_ptcl_per_obj = 20
    donut_radius = n_ptcl_per_obj * 4./3. * np.pi * radius**3 / (np.pi * radius**2 * 2 * np.pi)
    outer_diamter = 2 * (donut_radius + radius)

    # spacing = np.linspace(50 - 5 * outer_diamter, 50 + 5 * outer_diamter, n_objects_axis)
    # spacing = np.linspace(25 - 2.1 * outer_diamter, 25 + 2.1 * outer_diamter, n_objects_axis)
    spacing = np.linspace(22 - 1 * outer_diamter, 22 + 1 * outer_diamter, n_objects_axis)


    n_objects = len(spacing)**3 
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / n_ptcl_per_obj)
    ptcl_arm_ref_each = np.vstack((donut_radius * np.cos(angles), donut_radius * np.sin(angles), np.zeros(n_ptcl_per_obj))).T
    ptcl_arm_ref = list(np.repeat(ptcl_arm_ref_each[None, ...], n_objects, axis=0))
    radii_list = jax.tree_map(lambda x: radius*np.ones((x.shape[0])), ptcl_arm_ref)
 
    obj_x0 = np.stack(np.meshgrid(*([spacing]*3), indexing='ij')).reshape(dim, -1).T
    key = jax.random.PRNGKey(0)
    obj_q0 = jax.random.normal(key, (n_objects, dim + 1))
    obj_q0 /= np.linalg.norm(obj_q0, axis=1)[:, None]
    obj_y0 = np.hstack((obj_x0, obj_q0, np.zeros((n_objects, 6))))

    n_rope = 50
    rope_x0 = np.linspace(30, 30 + (n_rope - 5) * 2 * radius, n_rope)
    rope_y0 = np.hstack((rope_x0[:, None], 70. * np.ones((n_rope, 2)), np.ones((n_rope, 1)), np.zeros((n_rope, 9))))

    y0 = np.vstack((obj_y0, rope_y0))
    # y0 = obj_y0

    obj_rope_split = n_objects
    ptcl_rope_split = n_objects * n_ptcl_per_obj
    rope_radii = radius * np.ones(n_rope)

    nondiff_kwargs = {'normal_contact_stiffness': 1e4,
                      'damping_coeff': 1e1,
                      'Coulomb_fric_coeff': 0.5,
                      'tangent_fric_coeff': 1e1,
                      'rolling_fric_coeff': 0.2,
                      'box_env_bottom': 10., 
                      'box_env_top': 90.,
                      'ptcl_arm_ref_arr': np.stack(ptcl_arm_ref),
                      'radii_arr': np.stack(radii_list),
                      'obj_rope_split': obj_rope_split,
                      'ptcl_rope_split': ptcl_rope_split,
                      'rope_radii': rope_radii}

    env_bottom = nondiff_kwargs['box_env_bottom']
    env_top = nondiff_kwargs['box_env_top']

    def fwd_sim():
        dt = 2 * 1e-3
        ts = np.arange(0., dt*3001, dt)
        diff_kwargs = {}
        state_rhs_func = get_state_rhs_func(obj_state_rhs_func_prm_uniform, tuple(diff_kwargs.keys()), box_env, nondiff_kwargs)

        ys = simulate([y0, tuple(diff_kwargs.values())], ts, state_rhs_func)
        ys_skip = ys[::20]
        np.save(f'data/numpy/vedo/states_{case_name}.npy', ys_skip)

        # obj_ys_skip = np.load(f'data/numpy/vedo/states_{case_name}.npy')

        obj_ys_skip = ys_skip[:, :obj_rope_split, :]
        ptcl_ys_skip = obj_states_to_ptcl_states(obj_ys_skip, nondiff_kwargs['ptcl_arm_ref_arr'])

        rope_ys_skip = ys_skip[:, obj_rope_split:, :]
        joint_ys_skip = np.concatenate((ptcl_ys_skip, rope_ys_skip), axis=1)

        # energy = test_energy(ptcl_ys_skip, nondiff_kwargs['radii_arr'].reshape(-1))
        # plot_energy(energy, f'data/pdf/energy_{case_name}.pdf')
        vedo_plot(case_name, radius, env_bottom, env_top, joint_ys_skip, ptcl_rope_split)

    fwd_sim()



@walltime
def tetra():
    case_name = 'tetra'

    radius = 0.5

    n_objects_axis = 5
    spacing_x = np.linspace(30, 70, n_objects_axis - 1)
    spacing_y = np.linspace(30, 70, n_objects_axis - 1)
    spacing_z = np.linspace(50, 50, 1)

    n_row = 3
    n_col = 5
    base_x = 30.
    base_y = 30.
    base_z = 80.
    space_x = 8.
    space_y = 8.
    obj_y0 = []
    for i in range(n_row):
        for j in range(n_col):
            x = np.array([base_x + i * space_x, base_y + j * space_y, base_z])

            q1 = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0., 0.])
            q2 = np.array([np.cos(np.pi/4), 0., 0., np.sin(np.pi/4)])
            if (i + j) % 2 == 0:
                q = q1
            else:
                q = quat_mul(q2, q1)
            state = np.hstack((x, q, np.zeros(6)))
            obj_y0.append(state)

    obj_y0 = np.stack(obj_y0)
    n_objects = obj_y0.shape[0]

    # n_rope = 50
    # cmp_ratio = 1.
    # rope_length = (n_rope - 1) * 2 * radius * cmp_ratio
    # rope_x = np.linspace(base_y - 5, base_y - 5 + rope_length, n_rope)
    # rope_y0 = np.hstack((base_x * np.ones((n_rope, 1)), rope_x[:, None], base_z * np.ones((n_rope, 1)), np.ones((n_rope, 1)), np.zeros((n_rope, 9))))


    def seg_helper(start, seg_len, x_or_y):
        if x_or_y == 'x':
            seg_x = np.arange(start[0], start[0] + seg_len, 2*radius * np.sign(seg_len))
            seg_y = start[1] * np.ones_like(seg_x)
        else:
            seg_y = np.arange(start[1], start[1] + seg_len, 2*radius * np.sign(seg_len))
            seg_x = start[0] * np.ones_like(seg_y)
        return seg_x, seg_y

    rope_x = np.array([])
    rope_y = np.array([])
    for i in range(n_row):
        if i % 2 == 0:
            y1 = base_y - 0.5 * space_y
            y2 = base_y + (n_col - 0.5) * space_y
            y_len = n_col * space_y
        else:
            y1 = base_y + (n_col - 0.5) * space_y
            y2 = base_y - 0.5 * space_y
            y_len = -n_col * space_y           

        seg1_x, seg1_y = seg_helper([base_x + (i - 0.5) * space_x, y1], 0.5 * space_x, 'x')
        seg2_x, seg2_y = seg_helper([base_x + i * space_x, y1], y_len , 'y')
        seg3_x, seg3_y = seg_helper([base_x + i * space_x, y2], 0.5 * space_x, 'x')
        rope_x = np.hstack((rope_x, seg1_x, seg2_x, seg3_x))
        rope_y = np.hstack((rope_y, seg1_y, seg2_y, seg3_y))

    for j in range(n_col):
        if j % 2 == 0:
            x1 = base_x + (n_row - 0.5) * space_x
            x2 = base_x - 0.5 * space_x
            x_len = -n_row * space_x
        else:
            x1 = base_x - 0.5 * space_x
            x2 = base_x + (n_row - 0.5) * space_x
            x_len = n_row * space_x

        seg1_x, seg1_y = seg_helper([x1, base_y + (n_col - j - 0.5) * space_y], -0.5 * space_y, 'y')
        seg2_x, seg2_y = seg_helper([x1, base_y + (n_col - j - 1) * space_y], x_len , 'x')
        seg3_x, seg3_y = seg_helper([x2, base_y + (n_col - j - 1) * space_y], -0.5 * space_y, 'y')     
        rope_x = np.hstack((rope_x, seg1_x, seg2_x, seg3_x))
        rope_y = np.hstack((rope_y, seg1_y, seg2_y, seg3_y))

    rope_z = base_z * np.ones_like(rope_x)
    n_rope = len(rope_x)
    rope_x_ = np.stack((rope_x, rope_y, rope_z)).T

    rope_y0 = np.hstack((rope_x_, np.ones((n_rope, 1)), np.zeros((n_rope, 9))))

    y0 = np.vstack((obj_y0, rope_y0))
    # y0 = obj_y0

    # print(obj_y0)
    # exit()

    ptcl_arm_ref_each = np.load(f"data/numpy/convert/hollow_tetra.npy")
    n_ptcl_per_obj = len(ptcl_arm_ref_each)
    ptcl_arm_ref = list(np.repeat(ptcl_arm_ref_each[None, ...], n_objects, axis=0))
    radii_list = jax.tree_map(lambda x: radius*np.ones((x.shape[0])), ptcl_arm_ref)

    obj_rope_split = n_objects    
    ptcl_rope_split = n_objects * n_ptcl_per_obj
    rope_radii = radius * np.ones(n_rope)

    nondiff_kwargs = {'normal_contact_stiffness': 1e5,
                      'damping_coeff': 1e2,
                      'Coulomb_fric_coeff': 0.5,
                      # 'tangent_fric_coeff': 1e1,
                      'tangent_fric_coeff': 1e2,
                      'rolling_fric_coeff': 0.2,
                      'box_env_bottom': 10., 
                      'box_env_top': 90.,
                      'ptcl_arm_ref_arr': np.stack(ptcl_arm_ref),
                      'radii_arr': np.stack(radii_list),
                      'obj_rope_split': obj_rope_split,
                      'ptcl_rope_split': ptcl_rope_split,
                      'rope_radii': rope_radii}

    env_bottom = nondiff_kwargs['box_env_bottom']
    env_top = nondiff_kwargs['box_env_top']

    def fwd_sim():
        dt = 2 * 1e-3
        # dt = 1e-3
        ts = np.arange(0., dt*10001, dt)
        diff_kwargs = {}
        state_rhs_func = get_state_rhs_func(obj_state_rhs_func_prm_uniform, tuple(diff_kwargs.keys()), box_env, nondiff_kwargs)

        ys = simulate([y0, tuple(diff_kwargs.values())], ts, state_rhs_func)
        ys_skip = ys[::20]
        np.save(f'data/numpy/vedo/states_{case_name}.npy', ys_skip)

        # obj_ys_skip = np.load(f'data/numpy/vedo/states_{case_name}.npy')

        obj_ys_skip = ys_skip[:, :obj_rope_split, :]
        ptcl_ys_skip = obj_states_to_ptcl_states(obj_ys_skip, nondiff_kwargs['ptcl_arm_ref_arr'])

        rope_ys_skip = ys_skip[:, obj_rope_split:, :]
        joint_ys_skip = np.concatenate((ptcl_ys_skip, rope_ys_skip), axis=1)

        print(joint_ys_skip[-1])

        # energy = test_energy(ptcl_ys_skip, nondiff_kwargs['radii_arr'].reshape(-1))
        # plot_energy(energy, f'data/pdf/energy_{case_name}.pdf')
        vedo_plot(case_name, radius, env_bottom, env_top, joint_ys_skip, ptcl_rope_split)

    fwd_sim()



if __name__ == '__main__':
    # single_bouncing_ball()
    # billiards()
    donut_with_rope()
    # objects_in_drum()
    # donuts()
    # tetra()
    # plt.show()
    