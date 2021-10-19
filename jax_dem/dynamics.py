import jax
import jax.numpy as np
import numpy as onp
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import vedo
from functools import partial
from scipy.spatial.transform import Rotation as R
from jax_dem.utils import quats_mul, quat_mul, rotate_vector, rotate_vector_batch
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
    ptcl_arm_ref_list: np.ndarray = None
    radii_list: np.ndarray = None
    ptcl_split: np.ndarray = None
    ptcl_arm_ref_arr: np.ndarray = None
    radii_arr: np.ndarray = None


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


def compute_sphere_inertia_tensors(radii):
    vols = 4./3.*np.pi*radii**3
    inertias = 2./5.*vols*radii**2
    inertias = inertias[:, None, None] * np.eye(dim)[None, :, :]
    return inertias, vols


def get_unit_vectors(vectors):
    # norms = np.sqrt(np.sum(vectors**2, axis=-1))
    norms = norm(vectors)
    norms_reg = np.where(norms == 0., 1., norms)
    unit_vectors = vectors / norms_reg[..., None]
    return norms, unit_vectors



# def compute_forces_helper(parameter, intersected_distances, unit_vectors, relative_v, reduced_mass):

#     Coulomb_fric_coeff = parameter.Coulomb_fric_coeff
#     normal_contact_stiffness = parameter.normal_contact_stiffness
#     damping_coeff = parameter.damping_coeff
#     tangent_fric_coeff = parameter.tangent_fric_coeff

#     elastic_normal_forces = -normal_contact_stiffness * np.where(intersected_distances > 0., intersected_distances, 0.)[..., None] * unit_vectors

#     normal_velocity = np.sum(relative_v * unit_vectors, axis=-1)[..., None] * unit_vectors
#     damping_forces = 2 * damping_coeff * np.where(intersected_distances > 0., reduced_mass, 0.)[..., None] * normal_velocity

#     tangent_velocity = relative_v - normal_velocity
#     friction = 2 * tangent_fric_coeff * reduced_mass[..., None] * tangent_velocity
#     friction_norms, friction_unit_vectors = get_unit_vectors(friction)
#     friction_bounds = Coulomb_fric_coeff * norm(elastic_normal_forces)
#     friction_forces = np.where(friction_norms < friction_bounds, friction_norms, friction_bounds)[..., None] * friction_unit_vectors

#     forces = elastic_normal_forces + damping_forces + friction_forces

#     return forces
  

def compute_reactions_helper(parameter, intersected_distances, unit_vectors, relative_v, relative_w, reduced_mass, reduced_radius, arms):
    rolling_fric_coeff = parameter.rolling_fric_coeff

    Coulomb_fric_coeff = parameter.Coulomb_fric_coeff
    normal_contact_stiffness = parameter.normal_contact_stiffness
    damping_coeff = parameter.damping_coeff
    tangent_fric_coeff = parameter.tangent_fric_coeff

    elastic_normal_forces = -normal_contact_stiffness * np.where(intersected_distances > 0., intersected_distances, 0.)[..., None] * unit_vectors

    normal_velocity = np.sum(relative_v * unit_vectors, axis=-1)[..., None] * unit_vectors
    damping_forces = 2 * damping_coeff * np.where(intersected_distances > 0., reduced_mass, 0.)[..., None] * normal_velocity

    tangent_velocity = relative_v - normal_velocity
    friction = 2 * tangent_fric_coeff * reduced_mass[..., None] * tangent_velocity
    friction_norms, friction_unit_vectors = get_unit_vectors(friction)
    friction_bounds = Coulomb_fric_coeff * norm(elastic_normal_forces)
    friction_forces = np.where(friction_norms < friction_bounds, friction_norms, friction_bounds)[..., None] * friction_unit_vectors

    forces = elastic_normal_forces + damping_forces + friction_forces

    # forces = compute_forces_helper(parameter, intersected_distances, unit_vectors, relative_v, reduced_mass) 

    _, relative_w_unit_vectors = get_unit_vectors(relative_w)
    rolling_torque = rolling_fric_coeff * norm(elastic_normal_forces + damping_forces)[..., None] * reduced_radius[..., None] * relative_w_unit_vectors

    torques = np.cross(arms, forces) + rolling_torque

    return forces, torques


# @jax.jit
# @partial(jax.jit, static_argnums=(3,))
# def state_rhs_func_prm(state, t, parameter, env):

#     radii = parameter.radii

#     n_objects = state.shape[1]
#     x = state[0:3].T
#     q = state[3:7].T
#     v = state[7:10].T
#     w = state[10:13].T
#     inertias, vol = compute_sphere_inertia_tensors(radii)

#     box_size = onp.array([100., 100., 100.])
#     cell_capacity = 5
#     minimum_cell_size = 1.

#     cell_id, indices = cell_fn(x, box_size, minimum_cell_size, cell_capacity)

#     neighour_indices = tuple(indices_1_to_27(indices))

#     neighour_ids = cell_id[neighour_indices].reshape(n_objects, -1) # (n_objects, dim**3 * cell_capacity)

#     neighour_ids = prune_neighbour_list(x, radii, neighour_ids)

#     neighour_x = x[neighour_ids]
#     neighour_v = v[neighour_ids]
#     neighour_w = w[neighour_ids]
#     neighour_radii = radii[neighour_ids]
#     neighour_vol = vol[neighour_ids]

#     mutual_vectors = neighour_x - x[:, None, :]
#     mutual_distances,  mutual_unit_vectors = get_unit_vectors(mutual_vectors)
#     mutual_intersected_distances = neighour_radii + radii[:, None] - mutual_distances
#     mutual_contact_points = x[:, None, :] + (radii[:, None] - mutual_intersected_distances / 2.)[:, :, None] * mutual_unit_vectors
#     mutual_arms_self = mutual_contact_points - x[:, None, :]
#     mutual_arms_other = mutual_contact_points - neighour_x
#     mutual_relative_v = neighour_v + np.cross(neighour_w, mutual_arms_other) - v[:, None, :] - np.cross(w[:, None, :], mutual_arms_self)
#     mutual_relative_w = neighour_w - w[:, None, :] 
#     mutual_reduced_mass = neighour_vol * vol[:, None] / (neighour_vol + vol[:, None])
#     mutual_reduced_radius = neighour_radii * radii[:, None] / (neighour_radii + radii[:, None])

#     mutual_forces, mutual_torques = compute_reactions_helper(parameter, mutual_intersected_distances, mutual_unit_vectors, mutual_relative_v, 
#         mutual_relative_w, mutual_reduced_mass, mutual_reduced_radius, mutual_arms_self)

#     mask = np.logical_or(neighour_ids == n_objects, neighour_ids == np.arange(n_objects)[:, None])[:, :, None]
#     mutual_forces = np.where(mask, 0., mutual_forces)
#     mutual_torques = np.where(mask, 0., mutual_torques)

#     mutual_reactions = np.concatenate((np.sum(mutual_forces, axis=1), np.sum(mutual_torques, axis=1)), axis=-1)


#     env_distance_value, env_velocity = env(parameter)
#     env_distance_values = jax.vmap(env_distance_value, in_axes=0, out_axes=0)
#     env_distance_grad = jax.grad(env_distance_value, argnums=0)
#     env_distance_grads = jax.vmap(env_distance_grad, in_axes=0, out_axes=0)
#     env_velocities = jax.vmap(env_velocity, in_axes=0, out_axes=0)

#     env_intersected_distances = radii - env_distance_values(x)
#     env_unit_vectors = -env_distance_grads(x)
#     env_contact_points = x + (radii - env_intersected_distances / 2.)[:, None] * env_unit_vectors
#     env_arms = env_contact_points - x
#     env_relative_v = env_velocities(env_contact_points) - v - np.cross(w, env_arms)
#     env_relative_w = -w
#     env_reduced_mass = vol
#     env_reduced_radius = radii

#     env_forces, env_torques = compute_reactions_helper(parameter, env_intersected_distances, env_unit_vectors, env_relative_v, 
#         env_relative_w, env_reduced_mass, env_reduced_radius, env_arms)
#     env_reactions = np.concatenate((env_forces, env_torques), axis=-1)

#     contact_reactions = mutual_reactions + env_reactions

#     dx_rhs = v
#     w_quat = np.concatenate([np.zeros((1, n_objects)), w.T], axis=0)
#     dq_rhs = 0.5 * quats_mul(w_quat.T, q)
#     contact_forces = contact_reactions[:, :dim]
#     dv_rhs = (contact_forces / vol[:, None] + np.array([[0., 0., -parameter.gravity]]))
#     contact_torques = contact_reactions[:, dim:]
#     wIw = np.cross(w, np.squeeze(inertias @  w[..., None]))
#     I_inv = np.linalg.inv(inertias) 
#     dw_rhs = np.squeeze((I_inv @ (contact_torques - wIw)[..., None]), axis=-1)
#     rhs = np.concatenate([dx_rhs, dq_rhs, dv_rhs, dw_rhs], axis=1).T

#     particle_out_of_env = env_distance_values(x) < 0
#     cell_overflow = np.sum(cell_id != n_objects) != n_objects
#     # print(f"particle out of environment? {np.any(particle_out_of_env)}")
#     # print(f"cell overflow? {np.sum(cell_id != n_objects)}")
#     assert_condition = np.logical_or(particle_out_of_env, cell_overflow)
#     rhs = np.where(assert_condition, np.nan, rhs)
   
#     return rhs



def ptcl_state_rhs_func_prm(state, t, parameter, env):
    radii = parameter.radii

    ptcl_x = state[:, 0:3]
    ptcl_q = state[:, 3:7]
    ptcl_v = state[:, 7:10]
    ptcl_w = state[:, 10:13]
    inertias, vols = compute_sphere_inertia_tensors(radii)

    mutual_forces, mutual_torques, env_forces, env_torques, \
    _, _ = helper_partition(ptcl_x, ptcl_v, ptcl_w, parameter, env, radii, vols)

    torques = np.sum(mutual_torques, axis=1) + env_torques
    forces = np.sum(mutual_forces, axis=1) + env_forces + np.array([[0., 0., -parameter.gravity]]) * vols[:, None]
    rhs = compute_rhs(ptcl_q, ptcl_v, ptcl_w, forces, torques, inertias, vols)

    return rhs


def compute_object_inertia_tensor(xs, radii):
    inertias, vols = compute_sphere_inertia_tensors(radii)
    object_vol = np.sum(vols)
    object_centroid = np.sum(xs * vols[:, None], axis=0) / object_vol

    def parallel_axis_theorem(x, vol, inertia):
        x_rel = x - object_centroid
        return inertia + vol * (np.dot(x_rel, x_rel) * np.eye(dim) - np.outer(x_rel, x_rel))

    batch_parallel_axis_theorem = jax.vmap(parallel_axis_theorem, in_axes=(0, 0, 0), out_axes=0)
    object_inertia = np.sum(batch_parallel_axis_theorem(xs, vols, inertias), axis=0)
    return object_inertia, object_vol, object_centroid

compute_object_inertia_tensor_batch = jax.vmap(compute_object_inertia_tensor, in_axes=(0, 0), out_axes=(0, 0, 0))



def helper_partition(x, v, w, parameter, env, radii, vols):
    n_particles = x.shape[0]

    box_size = onp.array([100., 100., 100.])
    cell_capacity = 5
    minimum_cell_size = 1.


    cell_id, indices = cell_fn(x, box_size, minimum_cell_size, cell_capacity)
    neighour_indices = tuple(indices_1_to_27(indices))
    neighour_ids = cell_id[neighour_indices].reshape(n_particles, -1) # (n_particles, dim**3 * cell_capacity)
    neighour_ids = prune_neighbour_list(x, radii, neighour_ids)

    neighour_x = x[neighour_ids]
    neighour_v = v[neighour_ids]
    neighour_w = w[neighour_ids]
    neighour_radii = radii[neighour_ids]
    neighour_vols = vols[neighour_ids]

    mutual_vectors = neighour_x - x[:, None, :]
    mutual_distances,  mutual_unit_vectors = get_unit_vectors(mutual_vectors)
    mutual_intersected_distances = neighour_radii + radii[:, None] - mutual_distances
    mutual_contact_points = x[:, None, :] + (radii[:, None] - mutual_intersected_distances / 2.)[:, :, None] * mutual_unit_vectors
    mutual_arms_self = mutual_contact_points - x[:, None, :]
    mutual_arms_other = mutual_contact_points - neighour_x
    mutual_relative_v = neighour_v + np.cross(neighour_w, mutual_arms_other) - v[:, None, :] - np.cross(w[:, None, :], mutual_arms_self)
    mutual_relative_w = neighour_w - w[:, None, :] 
    mutual_reduced_mass = neighour_vols * vols[:, None] / (neighour_vols + vols[:, None])
    mutual_reduced_radius = neighour_radii * radii[:, None] / (neighour_radii + radii[:, None])

    # mutual_forces = compute_forces_helper(parameter, mutual_intersected_distances, mutual_unit_vectors, mutual_relative_v, mutual_reduced_mass)

    mutual_forces, mutual_torques = compute_reactions_helper(parameter, mutual_intersected_distances, mutual_unit_vectors, mutual_relative_v, 
        mutual_relative_w, mutual_reduced_mass, mutual_reduced_radius, mutual_arms_self)

    mask = np.logical_or(neighour_ids == n_particles, neighour_ids == np.arange(n_particles)[:, None])[:, :, None]

    mutual_forces = np.where(mask, 0., mutual_forces)
    mutual_torques = np.where(mask, 0., mutual_torques)

    env_distance_value, env_velocity = env(parameter)
    env_distance_values = jax.vmap(env_distance_value, in_axes=0, out_axes=0)
    env_distance_grad = jax.grad(env_distance_value, argnums=0)
    env_distance_grads = jax.vmap(env_distance_grad, in_axes=0, out_axes=0)
    env_velocities = jax.vmap(env_velocity, in_axes=0, out_axes=0)

    env_intersected_distances = radii - env_distance_values(x)
    env_unit_vectors = -env_distance_grads(x)
    env_contact_points = x + (radii - env_intersected_distances / 2.)[:, None] * env_unit_vectors
    env_arms = env_contact_points - x
    env_relative_v = env_velocities(env_contact_points) - v - np.cross(w, env_arms)
    env_relative_w = -w
    env_reduced_mass = vols
    env_reduced_radius = radii

    env_forces, env_torques = compute_reactions_helper(parameter, env_intersected_distances, env_unit_vectors, env_relative_v, 
        env_relative_w, env_reduced_mass, env_reduced_radius, env_arms)

    particle_out_of_env = env_distance_values(x) < 0
    cell_overflow = np.sum(cell_id != n_particles) != n_particles
    # print(f"particle out of environment? {np.any(particle_out_of_env)}")
    # print(f"cell overflow? {np.sum(cell_id != n_particles)}")
    assert_condition = np.logical_or(particle_out_of_env, cell_overflow)
    mutual_forces = np.where(assert_condition[:, None, None], np.nan, mutual_forces)

    return mutual_forces, mutual_torques, env_forces, env_torques, mutual_contact_points, env_contact_points

    # mutual_reactions = np.concatenate((np.sum(mutual_forces, axis=1), np.sum(mutual_torques, axis=1)), axis=-1)
    # env_reactions = np.concatenate((env_forces, env_torques), axis=-1)
    # contact_reactions = mutual_reactions + env_reactions


def compute_rhs(q, v, w, forces, torques, inertias, vols):
    dx_rhs = v
    w_quat = np.hstack((np.zeros((v.shape[0], 1)), w))
    dq_rhs = 0.5 * quats_mul(w_quat, q)
    dv_rhs = forces / vols[:, None]
    wIw = np.cross(w, np.squeeze(inertias @  w[..., None]))
    I_inv = np.linalg.inv(inertias) 
    dw_rhs = np.squeeze((I_inv @ (torques - wIw)[..., None]), axis=-1)
    rhs = np.hstack((dx_rhs, dq_rhs, dv_rhs, dw_rhs))
    return rhs


def obj_to_ptcl_nonuniform(obj_x, obj_q, obj_v, obj_w, ptcl_arm_ref_list):
    ptcl_per_object = jax.tree_map(lambda x: x.shape[0], ptcl_arm_ref_list)
    ptcl_arm_crt_list =  jax.tree_multimap(lambda x, y: rotate_vector(x, y), list(obj_q), ptcl_arm_ref_list)

    ptcl_x_list = jax.tree_multimap(lambda x, y: x[None, :] + y, list(obj_x), ptcl_arm_crt_list)
    ptcl_q_list = jax.tree_multimap(lambda x, y: np.tile(x, (y, 1)), list(obj_q), ptcl_per_object)
    ptcl_v_list = jax.tree_multimap(lambda x, y, z: x[None, :] + np.cross(y[None, :], z), list(obj_v), list(obj_w), ptcl_arm_crt_list)
    ptcl_w_list = jax.tree_multimap(lambda x, y: np.tile(x, (y, 1)), list(obj_w), ptcl_per_object)

    return ptcl_x_list, ptcl_q_list, ptcl_v_list, ptcl_w_list


def obj_state_rhs_func_prm_nonuniform(state, t, parameter, env):
    ptcl_arm_ref_list = parameter.ptcl_arm_ref_list
    radii_list = parameter.radii_list
    ptcl_split = parameter.ptcl_split

    obj_x = state[:, 0:3]
    obj_q = state[:, 3:7]
    obj_v = state[:, 7:10]
    obj_w = state[:, 10:13]

    ptcl_x_list, ptcl_q_list, ptcl_v_list, ptcl_w_list = obj_to_ptcl(obj_x, obj_q, obj_v, obj_w, ptcl_arm_ref_list)

    ptcl_x = np.vstack(ptcl_x_list)
    ptcl_v = np.vstack(ptcl_v_list)
    ptcl_w = np.vstack(ptcl_w_list)
    radii = np.hstack(radii_list)

    obj_info = jax.tree_multimap(lambda x, y: compute_object_inertia_tensor(x, y), ptcl_x_list, radii_list)
    obj_inertias, obj_vols, _ = jax.tree_multimap(lambda *xs: np.stack(xs), *obj_info)

    inertias, vols = compute_sphere_inertia_tensors(radii)
    mutual_forces, _, env_forces, _, mutual_contact_points, env_contact_points \
     = helper_partition(ptcl_x, ptcl_v, ptcl_w, parameter, env, radii, vols)

    mutual_forces_list = np.split(mutual_forces, ptcl_split)
    env_forces_list = np.split(env_forces, ptcl_split)
    mutual_contact_points_list = np.split(mutual_contact_points, ptcl_split)
    env_contact_points_list = np.split(env_contact_points, ptcl_split)

    obj_mutual_torques = np.vstack(jax.tree_multimap(lambda x, y, z: np.sum(np.cross(x - y[None, None, :], z), axis=(0, 1)), 
        mutual_contact_points_list, list(obj_x), mutual_forces_list))
    obj_env_torques = np.vstack(jax.tree_multimap(lambda x, y, z: np.sum(np.cross(x - y[None, :], z), axis=0), 
        env_contact_points_list, list(obj_x), env_forces_list))
    obj_torques = obj_mutual_torques + obj_env_torques

    obj_mutual_forces = np.stack(jax.tree_map(lambda x: np.sum(x, axis=(0, 1)), mutual_forces_list))
    obj_env_forces = np.stack(jax.tree_map(lambda x: np.sum(x, axis=0), env_forces_list))
    gravity_forces = np.array([[0., 0., -parameter.gravity]]) * obj_vols[:, None]
    obj_forces = obj_mutual_forces + obj_env_forces + gravity_forces

    rhs = compute_rhs(obj_q, obj_v, obj_w, obj_forces, obj_torques, obj_inertias, obj_vols)
   
    return rhs


def obj_to_ptcl_uniform(obj_x, obj_q, obj_v, obj_w, ptcl_arm_ref_arr):
    n_ptcl_per_obj = ptcl_arm_ref_arr.shape[1]
    ptcl_arm_crt_arr = rotate_vector_batch(obj_q, ptcl_arm_ref_arr)
    ptcl_x_arr = obj_x[:, None, :] + ptcl_arm_crt_arr
    ptcl_q_arr = np.repeat(obj_q[:, None, :], n_ptcl_per_obj, axis=1)
    ptcl_v_arr = obj_v[:, None, :] + np.cross(obj_w[:, None, :], ptcl_arm_crt_arr)
    ptcl_w_arr = np.repeat(obj_w[:, None, :], n_ptcl_per_obj, axis=1)

    return ptcl_x_arr, ptcl_q_arr, ptcl_v_arr, ptcl_w_arr

obj_to_ptcl_uniform_batch = jax.vmap(obj_to_ptcl_uniform, in_axes=(0, 0, 0, 0, None), out_axes=(0, 0, 0, 0))


def obj_state_rhs_func_prm_uniform(state, t, parameter, env):
    ptcl_arm_ref_arr = parameter.ptcl_arm_ref_arr
    radii_arr = parameter.radii_arr
    radii = radii_arr.reshape(-1)
    n_obj = ptcl_arm_ref_arr.shape[0]
    n_ptcl_per_obj = ptcl_arm_ref_arr.shape[1]

    obj_x = state[:, 0:3]
    obj_q = state[:, 3:7]
    obj_v = state[:, 7:10]
    obj_w = state[:, 10:13]

    # arr: (n_obj, n_ptcl_per_obj, ...)
    ptcl_x_arr, ptcl_q_arr, ptcl_v_arr, ptcl_w_arr = obj_to_ptcl_uniform(obj_x, obj_q, obj_v, obj_w, ptcl_arm_ref_arr)

    obj_inertias, obj_vols, _  = compute_object_inertia_tensor_batch(ptcl_x_arr, radii_arr)

    inertias, vols = compute_sphere_inertia_tensors(radii)
    mutual_forces, _, env_forces, _, mutual_contact_points, env_contact_points \
     = helper_partition(ptcl_x_arr.reshape(-1, dim), ptcl_v_arr.reshape(-1, dim), ptcl_w_arr.reshape(-1, dim), parameter, env, radii, vols)

    mutual_forces_arr = mutual_forces.reshape(n_obj, n_ptcl_per_obj, -1, dim)
    env_forces_arr = env_forces.reshape(n_obj, n_ptcl_per_obj, dim)
    mutual_contact_points_arr = mutual_contact_points.reshape(n_obj, n_ptcl_per_obj, -1, dim)
    env_contact_points_arr = env_contact_points.reshape(n_obj, n_ptcl_per_obj, dim)
   
    obj_mutual_torques = np.sum(np.cross(mutual_contact_points_arr - obj_x[:, None, None, :], mutual_forces_arr), axis=(1, 2))
    obj_env_torques = np.sum(np.cross(env_contact_points_arr - obj_x[:, None, :], env_forces_arr), axis=1)
    obj_torques = obj_mutual_torques + obj_env_torques

    obj_mutual_forces = np.sum(mutual_forces_arr, axis=(1, 2))
    obj_env_forces = np.sum(env_forces_arr, axis=1)
    gravity_forces = np.array([[0., 0., -parameter.gravity]]) * obj_vols[:, None]

    obj_forces = obj_mutual_forces + obj_env_forces + gravity_forces

    rhs = compute_rhs(obj_q, obj_v, obj_w, obj_forces, obj_torques, obj_inertias, obj_vols)
   
    return rhs


@jax.jit
def compute_energy(radii, state):
    x = state[0:3].T
    q = state[3:7].T
    v = state[7:10].T
    w = state[10:13].T
    inertias, vol = compute_sphere_inertia_tensors(radii)
    total_energy = 1./2. * np.sum(w * np.squeeze(inertias @ w[:, :, None])) + \
     np.sum(1./2. * vol * np.sum(v**2, axis=-1) + gravity * vol * x[:, 2])
    return total_energy


def get_state_rhs_func(state_rhs_func_prm, diff_keys, env, nondiff_kwargs):
    def state_rhs_func(state, t, *diff_args):
        parameter = Parameter(**nondiff_kwargs)
        kwargs = dict(zip(diff_keys, diff_args))
        parameter = dataclasses.replace(parameter, **kwargs)
        return state_rhs_func_prm(state, t, parameter, env)
    return jax.jit(state_rhs_func)

 
