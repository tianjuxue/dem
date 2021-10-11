import jax
import jax.numpy as np
import numpy as onp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import vedo
from functools import partial
from scipy.spatial.transform import Rotation as R
from jax_dem.utils import quats_mul, quat_mul
from jax_dem.arguments import args
from jax_dem.partition import cell_fn, indices_1_to_27, prune_neighbour_list
from jax_dem import dataclasses
 
dim = args.dim


@dataclasses.dataclass
class Parameter:
    gravity: np.float32 = 9.8
    radii: np.ndarray = None 
    normal_contact_stiffness: np.float32 = 1e4
    damping_coeff: np.float32 = 1e1
    Coulomb_fric_coeff: np.float32 = 0.5
    tangent_fric_coeff: np.float32 = 1e1
    rolling_fric_coeff: np.float32 = 0.2
    box_env_bottom: np.float32 = None
    box_env_top: np.float32 = None
    drum_env_omega: np.ndarray = None


def box_env(parameter):
    env_bottom = parameter.box_env_bottom
    env_top = parameter.box_env_top

    def box_distance_value(point):
        distances_to_walls = np.concatenate((point - env_bottom, env_top - point))
        return np.min(distances_to_walls)

    def box_velocity(point):
        return np.zeros(dim)

    return box_distance_value, box_velocity


def drum_env(parameter):
    drum_radius = 10.
    env_bottom = parameter.box_env_bottom
    env_top = parameter.box_env_top    
    drum_center = (env_top + env_bottom) / 2. * np.ones(dim)
    omega = parameter.drum_env_omega

    def drum_distance_value(point):
        return drum_radius - norm(point - drum_center)

    def drum_velocity(point):
        return np.cross(omega, point - drum_center)

    return drum_distance_value, drum_velocity


def norm(x):
    '''safe norm to avoid jax.grad yielding np.nan'''
    x = np.sum(x**2, axis=-1)
    safe_x = np.where(x > 0., x, 0.)
    return np.sqrt(safe_x)


def compute_sphere_inertia_tensors(radius, n_objects):
    vol = 4./3.*np.pi*radius**3
    inertia = 2./5.*vol*radius**2
    inertias = inertia[:, None, None] * np.eye(dim)[None, :, :]
    return inertias, vol


def get_unit_vectors(vectors):
    # norms = np.sqrt(np.sum(vectors**2, axis=-1))
    norms = norm(vectors)
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
    friction_bounds = Coulomb_fric_coeff * norm(elastic_normal_forces)
    friction_forces = np.where(friction_norms < friction_bounds, friction_norms, friction_bounds)[..., None] * friction_unit_vectors

    _, relative_w_unit_vectors = get_unit_vectors(relative_w)
    rolling_torque = rolling_fric_coeff * norm(elastic_normal_forces + damping_forces)[..., None] * reduced_radius[..., None] * relative_w_unit_vectors

    forces = elastic_normal_forces + damping_forces + friction_forces
    torques = np.cross(arms, forces) + rolling_torque

    return forces, torques


# @jax.jit
# @partial(jax.jit, static_argnums=(3,))
def state_rhs_func_prm(state, t, parameter, env):

    radii = parameter.radii

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


    Coulomb_fric_coeff = parameter.Coulomb_fric_coeff
    normal_contact_stiffness = parameter.normal_contact_stiffness
    damping_coeff = parameter.damping_coeff
    tangent_fric_coeff = parameter.tangent_fric_coeff
    rolling_fric_coeff = parameter.rolling_fric_coeff

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


    env_distance_value, env_velocity = env(parameter)
    env_distance_values = jax.vmap(env_distance_value, in_axes=0, out_axes=0)
    env_distance_grad = jax.grad(env_distance_value, argnums=0)
    env_distance_grads = jax.vmap(env_distance_grad, in_axes=0, out_axes=0)
    env_velocities = jax.vmap(env_velocity, in_axes=0, out_axes=0)

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
    dv_rhs = (contact_forces / vol[:, None] + np.array([[0., 0., -parameter.gravity]]))
    contact_torques = contact_reactions[:, dim:]
    wIw = np.cross(w, np.squeeze(inertias @  w[..., None]))
    I_inv = np.linalg.inv(inertias) 
    dw_rhs = np.squeeze((I_inv @ (contact_torques - wIw)[..., None]), axis=-1)
    rhs = np.concatenate([dx_rhs, dq_rhs, dv_rhs, dw_rhs], axis=1).T

    particle_out_of_env = env_distance_values(x) < 0
    cell_overflow = np.sum(cell_id != n_objects) != n_objects
    # print(f"particle out of environment? {np.any(particle_out_of_env)}")
    # print(f"cell overflow? {np.sum(cell_id != n_objects)}")
    assert_condition = np.logical_or(particle_out_of_env, cell_overflow)
    rhs = np.where(assert_condition, np.nan, rhs)
   
    return rhs


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


def get_state_rhs_func(diff_keys, env, nondiff_kwargs):
    def state_rhs_func(state, t, *diff_args):
        parameter = Parameter(**nondiff_kwargs)
        kwargs = dict(zip(diff_keys, diff_args))
        parameter = dataclasses.replace(parameter, **kwargs)
        return state_rhs_func_prm(state, t, parameter, env)
    return jax.jit(state_rhs_func)


# def runge_kutta_4(variable, rhs, dt):
#     y_0 = variable
#     k_0 = rhs(y_0)
#     k_1 = rhs(y_0 + dt/2 * k_0)
#     k_2 = rhs(y_0 + dt/2 * k_1)
#     k_3 = rhs(y_0 + dt * k_2)
#     k = 1./6. * (k_0 + 2. * k_1 + 2. * k_2 + k_3)
#     y_1 = y_0 + dt * k
#     return y_1


# def simulate_for(key):
    
#     state, radii = initialize_state_many_objects(key)


#     num_steps = 5000
#     # dt = 1e-3
#     dt = 5*1e-3

#     t = 0.
#     states = [state]
#     energy = []
#     for i in range(num_steps):
#         t += dt
#         rhs_func = lambda variable: jax.jit(state_rhs_func)(variable, t, radii)
#         state = runge_kutta_4(state, rhs_func, dt)
#         if i % 20 == 0:
#             e = compute_energy(radii, state)
#             print(f"\nstep {i}, total energy={e}, quaternion square sum: {np.sum(state[3:7]**2)}")
#             # print(f"state=\n{state}")
#             if np.any(np.isnan(state)):
#                 print(f"state=\n{state}")
#                 break
#             energy.append(e)
#             states.append(state)

#     states = np.array(states)
#     energy = np.array(energy)


#     return states


 
