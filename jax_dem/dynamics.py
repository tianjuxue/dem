import jax
import jax.numpy as np
import numpy as onp
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge
from jax.experimental.ode import odeint
import time
import matplotlib.pyplot as plt
import vedo
from scipy.spatial.transform import Rotation as R
from .utils import quats_mul, quat_mul
from .io import plot_energy, vedo_plot
from jax_dem.arguments import args
from jax_dem.partition import cell_fn, indices_1_to_27, prune_neighbour_list


env_top = args.env_top
env_bottom = args.env_bottom
dim = args.dim
gravity = args.dim


def norm(x):
    '''safe norm to avoid jax.grad yielding np.nan'''
    x = np.sum(x**2)
    safe_x = np.where(x > 0., x, 0.)
    return np.sqrt(safe_x)


def box_env():
    def box_distance_value(point):
        distances_to_walls = np.concatenate((point - env_bottom, env_top - point))
        return np.min(distances_to_walls)

    def box_velocity(point):
        return np.zeros(dim)


def drum_env():
    drum_radius = 10.
    drum_center = (env_top + env_bottom) / 2. * np.ones(dim)
    omega = np.array([4./25.*np.pi, 0., 0.])

    def drum_distance_value(point):
        return drum_radius - norm(point - drum_center)

    def drum_velocity(point):
        return np.cross(omega, point - drum_center)

    return drum_distance_value, drum_velocity


env_distance_value, env_velocity = drum_env()

env_distance_values = jax.vmap(env_distance_value, in_axes=0, out_axes=0)
env_distance_grad = jax.grad(env_distance_value, argnums=0)
env_distance_grads = jax.vmap(env_distance_grad, in_axes=0, out_axes=0)

env_velocities = jax.vmap(env_velocity, in_axes=0, out_axes=0)



def compute_sphere_inertia_tensors(radius, n_objects):
    vol = 4./3.*np.pi*radius**3
    inertia = 2./5.*vol*radius**2
    inertias = inertia[:, None, None] * np.eye(dim)[None, :, :]
    return inertias, vol


def get_unit_vectors(vectors):
    norms = np.sqrt(np.sum(vectors**2, axis=-1))
    norms_reg = np.where(norms == 0., 1., norms)
    unit_vectors = vectors / norms_reg[..., None]
    return norms, unit_vectors


def compute_reactions_helper(Coulomb_fric_coeff,
                             normal_contact_stiffness,
                             damping_coeff,
                             tangent_fric_coeff,
                             rolling_fric_coeff,
                             intersected_distances,
                             unit_vectors, 
                             relative_v, 
                             relative_w, 
                             reduced_mass, 
                             reduced_radius,
                             arms):
    elastic_normal_forces = -normal_contact_stiffness * np.where(intersected_distances > 0., intersected_distances, 0.)[..., None] * unit_vectors

    normal_velocity = np.sum(relative_v * unit_vectors, axis=-1)[..., None] * unit_vectors
    damping_forces = 2 * damping_coeff * np.where(intersected_distances > 0., reduced_mass, 0.)[..., None] * normal_velocity

    tangent_velocity = relative_v - normal_velocity
    friction = 2 * tangent_fric_coeff * reduced_mass[..., None] * tangent_velocity
    friction_norms, friction_unit_vectors = get_unit_vectors(friction)
    friction_bounds = Coulomb_fric_coeff * np.sqrt(np.sum(elastic_normal_forces**2, axis=-1))
    friction_forces = np.where(friction_norms < friction_bounds, friction_norms, friction_bounds)[..., None] * friction_unit_vectors

    _, relative_w_unit_vectors = get_unit_vectors(relative_w)
    rolling_torque = rolling_fric_coeff * np.linalg.norm(elastic_normal_forces + \
        damping_forces, axis=-1)[..., None] * reduced_radius[..., None] * relative_w_unit_vectors

    forces = elastic_normal_forces + damping_forces + friction_forces
    torques = np.cross(arms, forces) + rolling_torque

    return forces, torques



@jax.jit
def state_rhs_func(state, t, *args):
    radii, = args
    n_objects = state.shape[1]
    x = state[0:3].T
    q = state[3:7].T
    v = state[7:10].T
    w = state[10:13].T
    inertias, vol = compute_sphere_inertia_tensors(radii, n_objects)

    box_size = onp.array([100., 100., 100.])
    cell_capacity = 5
    minimum_cell_size = 1.

    cell_id, indices = cell_fn(x, box_size, minimum_cell_size, cell_capacity)

    neighour_indices = tuple(indices_1_to_27(indices))
    neighour_ids = cell_id[neighour_indices].reshape(n_objects, -1) # (n_objects, dim**3 * cell_capacity)

    neighour_ids = prune_neighbour_list(x, radii, neighour_ids)

    neighour_x = x[neighour_ids]
    neighour_v = v[neighour_ids]
    neighour_w = w[neighour_ids]
    neighour_radii = radii[neighour_ids]
    neighour_vol = vol[neighour_ids]

    Coulomb_fric_coeff = 0.5
    normal_contact_stiffness = 1e4
    damping_coeff = 1e1
    tangent_fric_coeff = 1e1
    rolling_fric_coeff = 0.2

    # Coulomb_fric_coeff = 0.
    # normal_contact_stiffness = 1e4
    # damping_coeff = 0.
    # tangent_fric_coeff = 0.
    # rolling_fric_coeff = 0.

    mutual_vectors = neighour_x - x[:, None, :]
    mutual_distances,  mutual_unit_vectors = get_unit_vectors(mutual_vectors)
    mutual_intersected_distances = neighour_radii + radii[:, None] - mutual_distances
    mutual_contact_points = x[:, None, :] + (radii[:, None] - mutual_intersected_distances / 2.)[:, :, None] * mutual_unit_vectors
    mutual_arms_self = mutual_contact_points - x[:, None, :]
    mutual_arms_other = mutual_contact_points - neighour_x
    mutual_relative_v = neighour_v + np.cross(neighour_w, mutual_arms_other) - v[:, None, :] - np.cross(w[:, None, :], mutual_arms_self)
    mutual_relative_w = neighour_w - w[:, None, :] 
    mutual_reduced_mass = neighour_vol * vol[:, None] / (neighour_vol + vol[:, None])
    mutual_reduced_radius = neighour_radii * radii[:, None] / (neighour_radii + radii[:, None])

    mutual_forces, mutual_torques = compute_reactions_helper(Coulomb_fric_coeff, normal_contact_stiffness, damping_coeff, 
        tangent_fric_coeff, rolling_fric_coeff, mutual_intersected_distances, mutual_unit_vectors, mutual_relative_v, 
        mutual_relative_w, mutual_reduced_mass, mutual_reduced_radius, mutual_arms_self)

    mask = np.logical_or(neighour_ids == n_objects, neighour_ids == np.arange(n_objects)[:, None])[:, :, None]
    mutual_forces = np.where(mask, 0., mutual_forces)
    mutual_torques = np.where(mask, 0., mutual_torques)

    mutual_reactions = np.concatenate((np.sum(mutual_forces, axis=1), np.sum(mutual_torques, axis=1)), axis=-1)


    env_intersected_distances = radii - env_distance_values(x)
    env_unit_vectors = -env_distance_grads(x)
    env_contac_points = x + (radii - env_intersected_distances / 2.)[:, None] * env_unit_vectors
    env_arms = env_contac_points - x
    env_relative_v = env_velocities(env_contac_points) - v - np.cross(w, env_arms)
    env_relative_w = -w
    env_reduced_mass = vol
    env_reduced_radius = radii

    env_forces, env_torques = compute_reactions_helper(Coulomb_fric_coeff, normal_contact_stiffness, damping_coeff, 
        tangent_fric_coeff, rolling_fric_coeff, env_intersected_distances, env_unit_vectors, env_relative_v, 
        env_relative_w, env_reduced_mass, env_reduced_radius, env_arms)
    env_reactions = np.concatenate((env_forces, env_torques), axis=-1)

    contact_reactions = mutual_reactions + env_reactions

    dx_rhs = v
    w_quat = np.concatenate([np.zeros((1, n_objects)), w.T], axis=0)
    dq_rhs = 0.5 * quats_mul(w_quat.T, q)
    contact_forces = contact_reactions[:, :dim]
    dv_rhs = (contact_forces / vol[:, None] + np.array([[0., 0., -gravity]]))
    contact_torques = contact_reactions[:, dim:]
    wIw = np.cross(w, np.squeeze(inertias @  w[..., None]))
    I_inv = np.linalg.inv(inertias) 
    dw_rhs = np.squeeze((I_inv @ (contact_torques - wIw)[..., None]), axis=-1)
    rhs = np.concatenate([dx_rhs, dq_rhs, dv_rhs, dw_rhs], axis=1).T

    cell_overflow = np.sum(cell_id != n_objects) != n_objects
    print(f"cell overflow? {cell_overflow}")
    rhs = np.where(cell_overflow, np.nan, rhs)

    return rhs


def runge_kutta_4(variable, rhs, dt):
    y_0 = variable
    k_0 = rhs(y_0)
    k_1 = rhs(y_0 + dt/2 * k_0)
    k_2 = rhs(y_0 + dt/2 * k_1)
    k_3 = rhs(y_0 + dt * k_2)
    k = 1./6. * (k_0 + 2. * k_1 + 2. * k_2 + k_3)
    y_1 = y_0 + dt * k
    return y_1


@jax.jit
def compute_energy(radii, state):
    x = state[0:3].T
    q = state[3:7].T
    v = state[7:10].T
    w = state[10:13].T
    inertias, vol = compute_sphere_inertia_tensors(radii, state.shape[1])
    total_energy = 1./2. * np.sum(w * np.squeeze(inertias @ w[:, :, None])) + \
     np.sum(1./2. * vol * np.sum(v**2, axis=-1) + gravity * vol * x[:, 2])
    return total_energy


def initialize_state_many_objects(key):
    radius = 0.5

    # spacing = np.linspace(env_bottom + 0.1*(env_top - env_bottom), env_top - 0.1*(env_top - env_bottom), 10)
    # n_objects_axis = 10
    # spacing = np.linspace(env_bottom + 2*radius, env_bottom + (4*n_objects_axis - 2)*radius, n_objects_axis)

    n_objects_axis = 10
    spacing = np.linspace(44.5, 55, n_objects_axis)

    n_objects = len(spacing)**3
    x1, x2, x3 = np.meshgrid(*([spacing]*3), indexing='ij')
    # key = jax.random.PRNGKey(0)
    perturb = jax.random.uniform(key, (dim, n_objects), np.float32, -0.5*radius, 0.5*radius)
    xx = np.concatenate([x1.reshape(1, -1), x2.reshape(1, -1), x3.reshape(1, -1)], axis=0)
    q0 = np.ones((1, n_objects))
    state = np.concatenate([xx, q0, np.zeros((9, n_objects))], axis=0)
    radii = radius * np.ones(state.shape[1])

    assert np.all(env_distance_values(xx.T) > 0), "Found particle outside of box!"

    return state, radii


def odeint_rk4(f, y0, t, *args):
    def step(state, t):
        y_prev, t_prev = state
        h = t - t_prev
        k1 = h * f(y_prev, t_prev, *args)
        k2 = h * f(y_prev + k1/2., t_prev + h/2., *args)
        k3 = h * f(y_prev + k2/2., t_prev + h/2., *args)
        k4 = h * f(y_prev + k3, t + h, *args)
        y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t), y

    _, ys = jax.lax.scan(step, (y0, t[0]), t[1:])


    # ys = []
    # state = (y0, t[0])
    # for time in t[1:]:
    #     state, y = step(state, time)
    #     ys.append(y)
    # ys = np.array(ys)

    return ys


# @odeint_rk4.defjvp
# def odeint_rk4_jvp(f, primals, tangents):
#   y0, t, *args = primals
#   delta_y0, _, *delta_args = tangents
#   nargs = len(args)

#   def f_aug(aug_state, t, *args_and_delta_args):
#     primal_state, tangent_state = aug_state
#     args, delta_args = args_and_delta_args[:nargs], args_and_delta_args[nargs:]
#     primal_dot, tangent_dot = jax.jvp(f, (primal_state, t, *args), (tangent_state, 0., *delta_args))
#     return np.stack([primal_dot, tangent_dot])

#   aug_init_state = np.stack([y0, delta_y0])
#   aug_states = odeint_rk4(f_aug, aug_init_state, t, *args, *delta_args)
#   ys, ys_dot = aug_states[:, 0, :], aug_states[:, 1, :]
#   return ys, ys_dot


# odeint_rk4 = jax.custom_jvp(odeint_rk4, nondiff_argnums=(0,))



def simulate_odeint(key):

    object_name = 'sparse_perfect_ball'
    dt = 1e-3
    ts = np.arange(0., 0.1, dt)
    y0, radii = initialize_state_many_objects(key)

    # states = odeint_rk4(state_rhs_func, y0, ts, radii)

    def objective(state_rhs_func, y0, ts, radii):
        states = odeint_rk4(state_rhs_func, y0, ts, radii)
        return np.sum(states)

    grad_obj = jax.grad(objective, argnums=-1)
    grads = grad_obj(state_rhs_func, y0, ts, radii)
    print(gras.shape)
    # print(gras[])

    exit()

    states = states[::20]

    print(f"Platform: {xla_bridge.get_backend().platform}")
    np.save(f'data/numpy/vedo/states_{object_name}.npy', states)
    # plot_energy(energy, f'data/pdf/energy_{object_name}.pdf')
    vedo_plot(object_name, radii, env_bottom, env_top, states)
    return states


def simulate_scan(key):
    object_name = 'sparse_perfect_ball'
    y0, radii = initialize_state_many_objects(key)
    
    dt = 1e-3
    ts = np.arange(0., 1., dt)

    def scan_fn(state, t):
        y_prev, t_prev = state
        rhs_func = lambda variable: state_rhs_func(variable, t_prev, radii) 
        y = runge_kutta_4(y_prev, rhs_func, dt)
        return (y, t), y

    _, ys = jax.lax.scan(scan_fn, (y0, ts[0]), ts[1:])

    # print(f"Platform: {xla_bridge.get_backend().platform}")
    # np.save(f'data/numpy/vedo/states_{object_name}.npy', ys)
    # plot_energy(energy, f'data/pdf/energy_{object_name}.pdf')
    # vedo_plot(object_name, radii, env_bottom, env_top,, ys)
    return ys


def simulate_for(key):
    object_name = 'sparse_perfect_ball'
    state, radii = initialize_state_many_objects(key)

    vedo_plot(object_name, radii, 40, 60)
    exit()

    num_steps = 5000
    # dt = 1e-3
    dt = 5*1e-3

    t = 0.
    states = [state]
    energy = []
    for i in range(num_steps):
        t += dt
        rhs_func = lambda variable: state_rhs_func(variable, t, radii)
        state = runge_kutta_4(state, rhs_func, dt)
        if i % 20 == 0:
            e = compute_energy(radii, state)
            print(f"\nstep {i}, total energy={e}, quaternion square sum: {np.sum(state[3:7]**2)}")
            # print(f"state=\n{state}")
            if np.any(np.isnan(state)):
                print(f"state=\n{state}")
                break
            energy.append(e)
            states.append(state)

    states = np.array(states)
    energy = np.array(energy)

    print(f"Platform: {xla_bridge.get_backend().platform}")
    np.save(f'data/numpy/vedo/states_{object_name}.npy', states)
    plot_energy(energy, f'data/pdf/energy_{object_name}.pdf')
    vedo_plot(object_name, radii, 40, 60, states)
    return states


def run():
    start_time = time.time()
    key = jax.random.PRNGKey(0)
    ys = simulate_for(key)
    end_time = time.time()
    print(f"Time elapsed {end_time-start_time}")   
    

if __name__ == '__main__':
    run()
