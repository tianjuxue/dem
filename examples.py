import numpy as onp
import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from jax.lib import xla_bridge
import time
from jax_dem.optimizer import optimize, simulate
from jax_dem.arguments import args
from jax_dem.dynamics import get_state_rhs_func, drum_env, box_env, drum_env, ptcl_state_rhs_func_prm, \
obj_state_rhs_func_prm_nonuniform, obj_state_rhs_func_prm_uniform, obj_to_ptcl_uniform_batch
from jax_dem.io import plot_energy, vedo_plot
import matplotlib.pyplot as plt

dim = args.dim


# TODO: transpose all states

def walltime(func):
    def wrapper():
        start_time = time.time()
        func()
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
        ts = np.arange(0., dt*501, dt)
        # ts = np.linspace(0., 8., 2001)
        # fig = plt.figure()
        omegas = np.array([4./25.*np.pi])
        # omegas = np.linspace(2./25.*np.pi, 26/25.*np.pi, 7)
        for i, omega in enumerate(omegas):
            print(f"Meta step {i}， omega = {omega}")
            diff_kwargs = {'drum_env_omega': np.array([omega, 0., 0.])}
            state_rhs_func = get_state_rhs_func(ptcl_state_rhs_func_prm, tuple(diff_kwargs.keys()), drum_env, nondiff_kwargs)
            ys = simulate([y0,  tuple(diff_kwargs.values())], ts, state_rhs_func)
            vedo_plot(case_name, nondiff_kwargs['radii'], env_bottom, env_top, ys[::20])

            # plt.plot(ts[1::20], -jax.vmap(obj_func)(ys[::20]), linestyle='-', marker='o', label=str(i))

        # plt.legend(fontsize=16, frameon=False, loc='upper right')
        # plt.show()

    def inv_opt():
        dt = 2*1e-3
        ts = np.arange(0., dt*1001, dt)
        # ts = np.linspace(0., 8., 2001)
        diff_kwargs = {'drum_env_omega': np.array([4./25.*np.pi, 0., 0.])}
        state_rhs_func = get_state_rhs_func(ptcl_state_rhs_func_prm, tuple(diff_kwargs.keys()), drum_env, nondiff_kwargs)
        lower_bound, _ = ravel_pytree([y0, np.array([0, 0., 0.])])
        upper_bound, _ = ravel_pytree([y0, np.array([100./25.*np.pi, 0., 0.])])
        bounds = onp.vstack([lower_bound, upper_bound]).T
        optimize([y0, tuple(diff_kwargs.values())], ts, obj_func, state_rhs_func, bounds)

    # fwd_sim()
    inv_opt()


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
        ys = simulate([y0, tuple(diff_kwargs.values())], ts, state_rhs_func)

        ptcl_x_arrs, ptcl_q_arrs, _, _ = obj_to_ptcl_uniform_batch(ys[:, :, 0:3], ys[:, :, 3:7], ys[:, :, 7:10], ys[:, :, 10:13], nondiff_kwargs['ptcl_arm_ref_arr'])
        states_plt = np.concatenate((ptcl_x_arrs.reshape(ys.shape[0], -1, dim), ptcl_q_arrs.reshape(ys.shape[0], -1, dim + 1)), axis=-1)[::20]
        np.save(f'data/numpy/vedo/states_{case_name}.npy', states_plt)
        vedo_plot(case_name, radius, env_bottom, env_top, states_plt)


    fwd_sim()


def donuts():
    case_name = 'donuts'

    radius = 0.5
    n_objects_axis = 10
    n_ptcl_per_obj = 20
    donut_radius = n_ptcl_per_obj * 4./3. * np.pi * radius**3 / (np.pi * radius**2 * 2 * np.pi)
    outer_diamter = 2 * (donut_radius + radius)
    spacing = np.linspace(50 - 5 * outer_diamter, 50 + 5 * outer_diamter, n_objects_axis)
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
                      'damping_coeff': 1e1,
                      'Coulomb_fric_coeff': 0.5,
                      'tangent_fric_coeff': 1e1,
                      'rolling_fric_coeff': 0.2,
                      'box_env_bottom': 10., 
                      'box_env_top': 90.,
                      'ptcl_arm_ref_arr': np.stack(ptcl_arm_ref),
                      'radii_arr': np.stack(radii_list)}

    env_bottom = nondiff_kwargs['box_env_bottom']
    env_top = nondiff_kwargs['box_env_top']

    def fwd_sim():
        dt = 2*1e-3
        ts = np.arange(0., dt*2001, dt)
        diff_kwargs = {'drum_env_omega': np.array([15./25.*np.pi, 0., 0.])}
        state_rhs_func = get_state_rhs_func(obj_state_rhs_func_prm_uniform, tuple(diff_kwargs.keys()), box_env, nondiff_kwargs)
        ys = simulate([y0, tuple(diff_kwargs.values())], ts, state_rhs_func)

        ptcl_x_arrs, ptcl_q_arrs, _, _ = obj_to_ptcl_uniform_batch(ys[:, :, 0:3], ys[:, :, 3:7], ys[:, :, 7:10], ys[:, :, 10:13], nondiff_kwargs['ptcl_arm_ref_arr'])
        states_plt = np.concatenate((ptcl_x_arrs.reshape(ys.shape[0], -1, dim), ptcl_q_arrs.reshape(ys.shape[0], -1, dim + 1)), axis=-1)[::20]
        np.save(f'data/numpy/vedo/states_{case_name}.npy', states_plt)
        vedo_plot(case_name, radius, env_bottom, env_top)


    fwd_sim()

if __name__ == '__main__':
    # single_bouncing_ball()
    # billiards()
    # particles_in_drum()
    # objects_in_drum()
    donuts()

