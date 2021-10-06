import numpy as onp
import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from jax.lib import xla_bridge
import time
from jax_dem.optimizer import optimize, simulate
from jax_dem.arguments import args
from jax_dem.dynamics import get_state_rhs_func, drum_env, box_env
from jax_dem.io import plot_energy, vedo_plot
import matplotlib.pyplot as plt

dim = args.dim


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
                      'box_env_top': 90., }

    # ts = np.arange(0., 1, 1e-2)
    ts = np.arange(0., 0.1, 0.2*1e-3)
    x0 = np.array([50, 50, nondiff_kwargs['box_env_bottom'] + 10*radius]).reshape(-1, 1)
    v0 = np.array([0, 0, -(9 + 4)*radius/0.1]).reshape(-1, 1)
    q0 = np.ones((1, n_objects))
    y0 = np.concatenate([x0, q0, np.zeros((3, n_objects)), v0, np.zeros((3, n_objects))], axis=0)

    def obj_func(yf):
        x = yf[0:3, :].T
        return x[0, 2]

    diff_kwargs = {} 
    diff_keys = tuple(diff_kwargs.keys())
    diff_args = tuple(diff_kwargs.values())
    state_rhs_func = get_state_rhs_func(diff_keys, box_env, nondiff_kwargs)

    bounds = onp.hstack([y0, y0])
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
                      'normal_contact_stiffness': 1e4,
                      'damping_coeff': 5,
                      'Coulomb_fric_coeff': 0.5,
                      'tangent_fric_coeff': 1e1,
                      'rolling_fric_coeff': 0.05,
                      'box_env_bottom': 10., 
                      'box_env_top': 90., }

    # ts = np.arange(0., 4., 1e-3)
    ts = np.arange(0., 0.5, 1e-3)
    # ts = np.arange(0., 800*1e-3, 1e-3)

    intersect = 4./3.*np.pi*radius**3 / nondiff_kwargs['normal_contact_stiffness']
    base_x = 50.
    base_y = 50.
    a = 1.05*radius
    b = np.sqrt(3.) * a
    x0_z = radius - intersect + nondiff_kwargs['box_env_bottom']
    x0 = np.array([[base_x, base_x, base_x-a, base_x+a, base_x-2*a, base_x,     base_x+2*a, base_x-3*a, base_x-a,   base_x+a,   base_x+3*a],
                   [48.,    base_y, base_y+b, base_y+b, base_y+2*b, base_y+2*b, base_y+2*b, base_y+3*b, base_y+3*b, base_y+3*b, base_y+3*b],
                   [x0_z,   x0_z,   x0_z,     x0_z,     x0_z,       x0_z,       x0_z,       x0_z,       x0_z,       x0_z,       x0_z      ]])
    v0 = np.hstack((np.array([[0., 10., 0.]]).T, np.zeros((dim, n_objects - 1))))
    q0 = np.ones((1, n_objects))
    y0 = np.concatenate([x0, q0, np.zeros((3, n_objects)), v0, np.zeros((3, n_objects))], axis=0)

    def obj_func(yf):
        xf = yf[0:2, -1]
        target_xf = np.array([base_x+3*a + 3*a, base_y+3*b + 3*b])
        return np.sum((xf - target_xf)**2)

    diff_kwargs = {} 
    diff_keys = tuple(diff_kwargs.keys())
    diff_args = tuple(diff_kwargs.values())
    state_rhs_func = get_state_rhs_func(diff_keys, box_env, nondiff_kwargs)

    lower_bound, _ = ravel_pytree([jax.ops.index_update(y0, jax.ops.index[(7, 8), (0, 0)], np.array([-50, -50])), diff_args])
    upper_bound, _ = ravel_pytree([jax.ops.index_update(y0, jax.ops.index[(7, 8), (0, 0)], np.array([50., 50.])), diff_args])

    bounds = onp.stack([lower_bound, upper_bound]).T

    # ys = simulate([y0, diff_args], ts, state_rhs_func)
    # ys_backward = simulate([ys[-1], diff_args], -ts[::-1], lambda state, t, *diff_args: -state_rhs_func(state, t, *diff_args))
    # print(y0)
    # print("\n")
    # print(ys[-1])
    # print("\n")
    # print(ys_backward[-1])
    # fig = plt.figure()
    # plt.plot(ts[-2::-1], ys_backward[:, 10, 0], linestyle='-', marker='o', color='red')
    # plt.plot(ts[1:], ys[:, 10, 0], linestyle='-', marker='o', color='blue')
    # plt.show()


    # np.save(f'data/numpy/vedo/states_{case_name}.npy', ys)
    # vedo_plot(case_name, nondiff_kwargs['radii'], nondiff_kwargs['box_env_bottom'], nondiff_kwargs['box_env_top'], ys_backward)

    optimize([y0, diff_args], ts, obj_func, state_rhs_func, bounds)
    # print(y0)


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
    y0 = np.concatenate([x0, q0, np.zeros((9, n_objects))], axis=0)
    radii = radius * np.ones(n_objects)

    diff_kwargs = {'radii': radii}
    diff_keys = tuple(diff_kwargs.keys())
    diff_args = tuple(diff_kwargs.values())
    state_rhs_func = get_state_rhs_func(diff_keys, box_env, nondiff_kwargs)
    ys = simulate([y0, diff_args], ts, state_rhs_func)
    
    # np.save(f'data/numpy/vedo/states_{case_name}.npy', ys)
    # vedo_plot(case_name, kwargs['radii'], env_bottom, env_top, states=None)


def particles_in_drum():
    omega = np.array([4./25.*np.pi, 0., 0.])
    n_objects_axis = 10
    spacing = np.linspace(44.5, 55, n_objects_axis)


if __name__ == '__main__':
    # single_bouncing_ball()
    billiards()
