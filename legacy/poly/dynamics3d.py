import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge
import time
import matplotlib.pyplot as plt
import vedo
from functools import partial
from scipy.spatial.transform import Rotation as R
from .arguments import args
from .shape3d import get_phy_seeds, compute_inertia_tensor, compute_inertia_tensors, quats_mul, generate_template_object, reference_to_physical, \
batch_eval_sdf, batch_grad_sdf, get_rot_mats, batch_reference_to_physical, quat_mul, get_ref_vertices_oriented, rotate_point, get_ref_seeds, \
batch_eval_sdf_helper, batch_grad_sdf_helper, batch_eval_sign
from .io import output_vtk_3D_shape, plot_energy
from .dyn_comms import batch_wall_eval_sdf, batch_wall_grad_sdf, get_frictionless_force, runge_kutta_4
import gc

# from jax.config import config
# config.update("jax_debug_nans", True)

dim = args.dim
gravity = args.gravity
box_size = args.box_size

# no_filter_double_vmap: no distance check, use double vmap (fastest for small number of objects, but for 125 objects, running on GPU asks for too much memoery)
# no_filter_lax: no distance check, use lax.map for outer loop, and vmap for inner loop (for 125 objects, very slow)
# filter_vmap: with distance check, use vmap for the collision pairs (relatively fast, but triggering recompiles when batch size changes)
# filter_lax-map: with distance check, use lax.map for the collision pairs (when collision pairs is empty, it is still slow, why?)
# filter_python-map: with distance check, use python map for collision pairs
# filter_alterntative: with distance check, use an alternative method that does not rely on SDF. Energy not conserved...
running_modes = ['no_filter_double_vmap', 'no_filter_lax-map_vmap', 'filter_vmap', 'filter_lax-map', 'filter_python-map', 'filter_alternative']

running_mode =  running_modes[2]


##########################################################################################
# Alternative ways to compute mutual reactions without using SDFs

def compute_mutual_reaction_vertices(params, directions, connectivity, ref_vertice_normals, ref_centroid, x, q, batch_phy_seeds, index1, index2):
    '''Force (f1, f2) and torque (t) by object 2 on object 1
    '''
    stiffness = 1e5
    x_o1 = x[index1]
    x_o2 = x[index2]    
    q_o1 = q[index1]
    q_o2 = q[index2] 
    phy_seeds_o1 = batch_phy_seeds[index1]
    phy_seeds_o2 = batch_phy_seeds[index2]

    def f3(_):
        # object 2 is master (use seeds of object 2), object 1 is slave
        phy_vertices_normals_o2 = rotate_point(q_o2, ref_vertice_normals)
        vertices_distances = get_mutual_distances(phy_seeds_o2, phy_seeds_o1)
        inds = np.argmin(vertices_distances, axis=0)
        closest_phy_seed_o2 = np.take(phy_seeds_o2, inds, axis=0)
        phy_vertices_normal_o2 = np.take(phy_vertices_normals_o2, inds, axis=0)
        d = np.sum(phy_vertices_normal_o2 * (closest_phy_seed_o2 - phy_seeds_o1), axis=1)
        forces = stiffness * np.where(d > 0, d, 0.).reshape(-1, 1) * phy_vertices_normal_o2
        reaction = get_reaction(closest_phy_seed_o2, x_o1, forces)
        return reaction

    def f2(_):
        # object 1 is master (use seeds of object 1), object 2 is slave
        phy_vertices_normals_o1 = rotate_point(q_o1, ref_vertice_normals)
        vertices_distances = get_mutual_distances(phy_seeds_o1, phy_seeds_o2)
        inds = np.argmin(vertices_distances, axis=0)
        closest_phy_seed_o1 = np.take(phy_seeds_o1, inds, axis=0)
        phy_vertices_normal_o1 = np.take(phy_vertices_normals_o1, inds, axis=0)
        tmp = phy_vertices_normal_o1 * (closest_phy_seed_o1 - phy_seeds_o2)
        d = np.sum(phy_vertices_normal_o1 * (closest_phy_seed_o1 - phy_seeds_o2), axis=1)
        forces = -stiffness * np.where(d > 0, d, 0.).reshape(-1, 1) * phy_vertices_normal_o1
        reaction = get_reaction(closest_phy_seed_o1, x_o1, forces)
        return reaction

    return jax.lax.cond(index1 < index2, f2, f3, None)

batch_compute_mutual_reaction_vertices = jax.jit(jax.vmap(compute_mutual_reaction_vertices, in_axes=(None,)*8 + (0,)*2, out_axes=0))


def compute_normals(params, directions, connectivity):
    ref_vertices_oriented, _ = get_ref_vertices_oriented(params, directions, connectivity)
    ED = ref_vertices_oriented[1] - ref_vertices_oriented[0]
    FD = ref_vertices_oriented[2] - ref_vertices_oriented[0]
    facet_normals = np.cross(ED, FD)
    facet_normals = facet_normals / np.linalg.norm(facet_normals, axis=-1).reshape(-1, 1)
    indices = connectivity.reshape(-1)
    normals = np.repeat(facet_normals, dim, axis=0)
    vertice_normals = groupby_mean(indices, normals, len(directions))
    vertice_normals = vertice_normals / np.linalg.norm(vertice_normals, axis=-1).reshape(-1, 1)
    return vertice_normals


def get_one_hot(indices, size_to):
    one_hot = indices.reshape(-1, 1) == np.arange(size_to).reshape(1, -1)
    return one_hot


def groupby_mean(indices, data, size_to):
    one_hot = get_one_hot(indices, size_to)
    return one_hot.T @ data / np.sum(one_hot, axis=0).reshape(-1, 1)


def groupby_sum(indices, data, size_to):
    one_hot = get_one_hot(indices, size_to)
    return one_hot.T @ data

##########################################################################################


@jax.jit
def get_reaction(phy_seeds, x, forces):
    toque_arms = phy_seeds - x.reshape(1, -1)
    torques = np.cross(toque_arms, forces)
    f = np.sum(forces, axis=0)
    t = np.sum(torques, axis=0)
    return np.concatenate([f, t])


def compute_wall_reaction(params, directions, connectivity, ref_centroid, x, q, phy_seeds):
    forces_left = get_frictionless_force(phy_seeds, partial(batch_wall_eval_sdf, 0., True, 0), partial(batch_wall_grad_sdf, 0., True, 0))
    reaction_left = get_reaction(phy_seeds, x, forces_left)
    forces_right = get_frictionless_force(phy_seeds, partial(batch_wall_eval_sdf, box_size, False, 0), partial(batch_wall_grad_sdf, box_size, False, 0))
    reaction_right = get_reaction(phy_seeds, x, forces_right)

    forces_front = get_frictionless_force(phy_seeds, partial(batch_wall_eval_sdf, 0., True, 1), partial(batch_wall_grad_sdf, 0., True, 1))
    reaction_front = get_reaction(phy_seeds, x, forces_front)
    forces_back = get_frictionless_force(phy_seeds, partial(batch_wall_eval_sdf, box_size, False, 1), partial(batch_wall_grad_sdf, box_size, False, 1))
    reaction_back = get_reaction(phy_seeds, x, forces_back)

    forces_bottom = get_frictionless_force(phy_seeds, partial(batch_wall_eval_sdf, 0., True, 2), partial(batch_wall_grad_sdf, 0., True, 2))
    reaction_bottom = get_reaction(phy_seeds, x, forces_bottom)
    forces_top = get_frictionless_force(phy_seeds, partial(batch_wall_eval_sdf, box_size, False, 2), partial(batch_wall_grad_sdf, box_size, False, 2))
    reaction_top = get_reaction(phy_seeds, x, forces_top)

    return reaction_left + reaction_right + reaction_front + reaction_back + reaction_bottom + reaction_top


batch_compute_wall_reaction = jax.jit(jax.vmap(compute_wall_reaction, in_axes=(None, None, None, None, 0, 0, 0), out_axes=0))



@jax.jit
def compute_sign(params, directions, connectivity, ref_centroid, x, q, batch_phy_seeds, index1, index2):
    '''Force (f1, f2) and torque (t) by object 2 on object 1
    '''
    x_o1 = x[index1]
    x_o2 = x[index2]    
    q_o1 = q[index1]
    q_o2 = q[index2] 
    phy_seeds_o1 = batch_phy_seeds[index1]
    phy_seeds_o2 = batch_phy_seeds[index2]
    num_seeds = len(phy_seeds_o1)

    def f3(_):
        # object 2 is master (use seeds of object 2), object 1 is slave
        signs = batch_eval_sign(params, directions, connectivity, ref_centroid, x_o1, q_o1, phy_seeds_o2)
        return signs

    def f2(_):
        # object 1 is master (use seeds of object 1), object 2 is slave
        signs = batch_eval_sign(params, directions, connectivity, ref_centroid, x_o2, q_o2, phy_seeds_o1)
        return jax.lax.cond(index1 < index2, lambda _:signs, f3, None)

    def f1(_):
        return np.ones((num_seeds,))

    return jax.lax.cond(index1 == index2, f1, f2, None)

compute_signs = jax.jit(jax.vmap(compute_sign, in_axes=(None,)*7 + (0,)*2, out_axes=0))



@jax.jit
def compute_mutual_reaction_sparse(params, directions, connectivity, ref_centroid, x, q, batch_phy_seeds, collision_indices, contact_index):
    '''Force (f1, f2) and torque (t) by object 2 on object 1
    '''
    index1 = collision_indices[contact_index[0]][0]
    index2 = collision_indices[contact_index[0]][1]
    seed_index = contact_index[1]

    x_o1 = x[index1]
    x_o2 = x[index2]    
    q_o1 = q[index1]
    q_o2 = q[index2] 
    phy_seeds_o1 = batch_phy_seeds[index1][seed_index].reshape(1, -1)
    phy_seeds_o2 = batch_phy_seeds[index2][seed_index].reshape(1, -1)

    def f3(_):
        # object 2 is master (use seeds of object 2), object 1 is slave
        level_set_func = partial(batch_eval_sdf, params, directions, connectivity, ref_centroid, x_o1, q_o1)
        level_set_grad = partial(batch_grad_sdf, params, directions, connectivity, ref_centroid, x_o1, q_o1)
        forces = -get_frictionless_force(phy_seeds_o2, level_set_func, level_set_grad)
        reaction = get_reaction(phy_seeds_o2, x_o1, forces)
        return reaction

    def f2(_):
        # object 1 is master (use seeds of object 1), object 2 is slave
        level_set_func = partial(batch_eval_sdf, params, directions, connectivity, ref_centroid, x_o2, q_o2)
        level_set_grad = partial(batch_grad_sdf, params, directions, connectivity, ref_centroid, x_o2, q_o2)
        forces = get_frictionless_force(phy_seeds_o1, level_set_func, level_set_grad)
        reaction = get_reaction(phy_seeds_o1, x_o1, forces)
        return jax.lax.cond(index1 < index2, lambda _:reaction, f3, None)

    def f1(_):
        return np.zeros((6,))

    return jax.lax.cond(contact_index[0] < 0, f1, f2, None)

bacth_compute_mutual_reaction_sparse = jax.jit(jax.vmap(compute_mutual_reaction_sparse, in_axes=(None,)*8 + (0,), out_axes=0))


@jax.jit
def compute_mutual_reaction_sdf(params, directions, connectivity, ref_centroid, x, q, batch_phy_seeds, index1, index2):
    '''Force (f1, f2) and torque (t) by object 2 on object 1
    '''
    x_o1 = x[index1]
    x_o2 = x[index2]    
    q_o1 = q[index1]
    q_o2 = q[index2] 
    phy_seeds_o1 = batch_phy_seeds[index1]
    phy_seeds_o2 = batch_phy_seeds[index2]

    def f3(_):
        # object 2 is master (use seeds of object 2), object 1 is slave
        level_set_func = partial(batch_eval_sdf, params, directions, connectivity, ref_centroid, x_o1, q_o1)
        level_set_grad = partial(batch_grad_sdf, params, directions, connectivity, ref_centroid, x_o1, q_o1)
        forces = -get_frictionless_force(phy_seeds_o2, level_set_func, level_set_grad)
        reaction = get_reaction(phy_seeds_o2, x_o1, forces)
        return reaction

    def f2(_):
        # object 1 is master (use seeds of object 1), object 2 is slave
        level_set_func = partial(batch_eval_sdf, params, directions, connectivity, ref_centroid, x_o2, q_o2)
        level_set_grad = partial(batch_grad_sdf, params, directions, connectivity, ref_centroid, x_o2, q_o2)
        forces = get_frictionless_force(phy_seeds_o1, level_set_func, level_set_grad)
        reaction = get_reaction(phy_seeds_o1, x_o1, forces)
        return jax.lax.cond(index1 < index2, lambda _:reaction, f3, None)

    def f1(_):
        return np.zeros((6,))

    return jax.lax.cond(index1 == index2, f1, f2, None)

batch_compute_mutual_reaction_sdf = jax.jit(jax.vmap(compute_mutual_reaction_sdf, in_axes=(None,)*7 + (0,)*2, out_axes=0))

batch_compute_mutual_reaction_sdf_index2 = jax.jit(jax.vmap(compute_mutual_reaction_sdf, in_axes=(None,)*8 + (0,)*1, out_axes=0))

batch_compute_mutual_reaction_sdf_index_1 = jax.jit(jax.vmap(batch_compute_mutual_reaction_sdf_index2, in_axes=(None,)*7 + (0, None), out_axes=0))


def add_to_target(target_index, reaction, target):
    return jax.ops.index_add(target, target_index, reaction)

reduce_at = jax.jit(jax.vmap(add_to_target, in_axes=(0, 0, None), out_axes=0))


def get_mutual_distances(pointsA, pointsB):
  return np.sqrt(np.sum((pointsA[:, None, :] - pointsB[None, :, :])**2, axis=-1))


def state_rhs_func(params, directions, connectivity, state):
    '''
    Parameter
    ---------
    params: numpy array of shape (n_params,)
    directions: numpy array with shape (num_vertices, dim)
    connectivity: numpy array with shape (num_cells, dim)
    state: numpy array of shape (13, n_objects)

    Returns
    -------
    rhs: numpy array of shape (13, n_objects)
    '''
    n_objects = state.shape[1]
    x = state[0:3].T
    q = state[3:7].T
    v = state[7:10].T
    w = state[10:13].T
    polyhedra_inertias, polyhedron_vol, ref_centroid = compute_inertia_tensors(params, directions, connectivity, q)
    I_inv = np.linalg.inv(polyhedra_inertias) 
    ref_seeds = get_ref_seeds(params, directions, connectivity)
    batch_phy_seeds = batch_reference_to_physical(x, q, ref_centroid, ref_seeds)

    break1 = time.time()

    if running_mode.startswith('no_filter'):
        # Do not check if two objects are far apart.
        if running_mode == 'no_filter_double_vmap': 
            paired_reactions = batch_compute_mutual_reaction_sdf_index_1(params, directions, connectivity, ref_centroid, 
                        x, q, batch_phy_seeds, np.arange(n_objects), np.arange(n_objects))

        if running_mode == 'no_filter_lax-map_vmap':
            def body_func(index):
                return batch_compute_mutual_reaction_sdf_index2(params, directions, connectivity, ref_centroid, 
                    x, q, batch_phy_seeds, index, np.arange(n_objects))
            paired_reactions = jax.lax.map(body_func, np.arange(n_objects))

        mutual_reactions = np.sum(paired_reactions, axis=1)

    if running_mode.startswith('filter'):
        # Check if two objects are far apart. If so, avoid computing mutual reactions.
        max_radius = np.max(params)
        phy_origins = batch_reference_to_physical(x, q, ref_centroid, np.array([0., 0., 0.]))
        mutual_distances = get_mutual_distances(phy_origins, phy_origins)
        collision_indices = np.array(np.where(mutual_distances < 2 * max_radius)).T
        collision_indices = collision_indices[np.where(collision_indices[:, 0] != collision_indices[:, 1])]

        def body_func(index_pair):
            return compute_mutual_reaction_sdf(params, directions, connectivity, ref_centroid, x, q, batch_phy_seeds, index_pair[0], index_pair[1])

        if running_mode == 'filter_vmap':
            # Add padding so that when collision_indices and contact_indices have fixed shapes
            # Example: collision_indices [[2, 3], [3, 2]] means object 2 and object 3 are in potential collision
            # contact_indices [[0, 1], [0, 2], [0, 5], [1, 1], [1, 2], [1, 5]] means object 2 (collision_indices[0][0])
            # and object 3 (collision_indices[0][1]) has seeds number 1, 2 and 5 in actual contact.

            n_bound_collision = 6 * n_objects
            n_bound_contact = 10 * n_objects
            collision_indices = np.concatenate([collision_indices, -1 * np.ones((n_bound_collision - len(collision_indices), 2), dtype=np.int32)], axis=0)

            signs = compute_signs(params, directions, connectivity, ref_centroid, x, q, batch_phy_seeds, collision_indices[:, 0], collision_indices[:, 1])
            contact_indices = np.array(np.where(signs < 0)).T

            contact_indices = np.concatenate([contact_indices, -1 * np.ones((n_bound_contact - len(contact_indices), 2), dtype=np.int32)], axis=0)
            reactions = bacth_compute_mutual_reaction_sparse(params, directions, connectivity, ref_centroid, x, q, batch_phy_seeds, 
                collision_indices, contact_indices)
            collision_indices = np.take(collision_indices, contact_indices[:, 0], axis=0)

            # reactions = batch_compute_mutual_reaction_sdf(params, directions, connectivity, 
            #     ref_centroid, x, q, batch_phy_seeds, collision_indices[:, 0], collision_indices[:, 1])

        if running_mode == 'filter_lax-map':
            reactions = jax.lax.map(body_func, collision_indices)

        if running_mode == 'filter_python-map':
            reactions = np.array(list(map(body_func, collision_indices)))
 
        if running_mode == 'filter_alternative':
            ref_vertice_normals = compute_normals(params, directions, connectivity)
            reactions = batch_compute_mutual_reaction_vertices(params, directions, connectivity, ref_vertice_normals, 
                ref_centroid, x, q, batch_phy_seeds, collision_indices[:, 0], collision_indices[:, 1])

        mutual_reactions = np.zeros((n_objects, 6))
        mutual_reactions = np.sum(reduce_at(collision_indices[:, 0], reactions, mutual_reactions), axis=0)

        # if len(collision_indices) > 1:
        #     print("Collision may happen...")

    break2 = time.time()

    wall_reactions = batch_compute_wall_reaction(params, directions, connectivity, ref_centroid, x, q, batch_phy_seeds)
    contact_reactions = mutual_reactions + wall_reactions

    dx_rhs = v.T

    w_quat = np.concatenate([np.zeros((1, n_objects)), w.T], axis=0)
    dq_rhs = 0.5 * quats_mul(w_quat.T, q).T

    contact_forces = contact_reactions[:, :3]
    dv_rhs = (contact_forces / polyhedron_vol + np.array([[0., 0., -gravity]])).T

    contact_torques = contact_reactions[:, 3:]
    M = np.expand_dims(contact_torques, axis=-1)
    wIw = np.expand_dims(np.cross(w, np.squeeze(polyhedra_inertias @  np.expand_dims(w, axis=-1))), axis=-1)
    dw_rhs = (I_inv @ (M - wIw)).reshape(n_objects, 3).T

    rhs = np.concatenate([dx_rhs, dq_rhs, dv_rhs, dw_rhs], axis=0)

    time_elapesed = break2 - break1 
    if time_elapesed > 1:
        print(f"---------------------------------------compute mutual reactions took {time_elapesed}s")
        # print(f"# of collision pairs = {len(collision_indices)}")
        # print(collision_indices)

    return rhs


def vedo_plot(object_name, ref_centroid=None, states=None):

    if ref_centroid is None:
        ref_centroid = np.load('data/numpy/vedo/ref_centroid.npy')

    if states is None:
        states = np.load('data/numpy/vedo/states.npy')

    n_objects = states.shape[-1]

    # vedo.settings.useDepthPeeling = False

    world = vedo.Box([box_size/2., box_size/2., box_size/2.], box_size, box_size, box_size).wireframe()

    stone = vedo.Mesh(f"data/vtk/3d/vedo/{object_name}.vtk").c("red").addShadow(z=0)
    stone.origin(*ref_centroid)
    stones = [stone.clone() for _ in range(n_objects)]

    vedo.show(world, *stones, axes=4, viewup="z", interactive=0)

    vd = vedo.Video(f"data/mp4/3d/{object_name}.mp4", fps=30)
    # Modify vd.options so that preview on Mac OS is enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    vd.options = "-b:v 8000k -pix_fmt yuv420p"

    for s in range(len(states) - 1):

        xx = states[s + 1][0:3]
        qq = states[s + 1][3:7]
        qq_prev = states[s][3:7]

        for i in range(n_objects):
            x = xx.T[i]
            q = qq.T[i]
            q_pre = qq_prev.T[i]
            q_inc = quat_mul(q, np.array([q_pre[0], -q_pre[1], -q_pre[2], -q_pre[3]]))
            r = R.from_quat([q_inc[1], q_inc[2], q_inc[3], q_inc[0]])
            rotvec = r.as_rotvec()
            angle = np.linalg.norm(rotvec)
            vec = rotvec/angle
            stones[i].pos(x).RotateWXYZ(angle * 180.0 / np.pi, vec[0], vec[1], vec[2])

        plotter = vedo.show(world, *stones, resetcam=False)

        print(f"frame: {s} in {len(states) - 1}")

        vd.addFrame()

    vd.close() 
    # vedo.interactive().close()


@jax.jit
def compute_energy(params, directions, connectivity, state):
    x = state[0:3]
    q = state[3:7]
    v = state[7:10]
    w = state[10:13]
    inertias, vol, _ = compute_inertia_tensors(params, directions, connectivity, q.T)
    total_energy = 1./2. * np.sum(w.T * np.squeeze(inertias @ np.expand_dims(w.T, axis=-1))) + 1./2. * vol * np.sum(v**2) + vol * gravity * np.sum(x[2])
    return total_energy
 
 
def initialize_state_1_object():
    state = np.array([10., 10., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(-1, 1)
    return state


def initialize_state_3_objects():
    state = np.array([[10., 10., 10.],
                      [10., 10., 10.],
                      # [2., 6, 10.],
                      # [1.1, 3.3, 10.],
                      [1.1, 3, 10.],
                      [1., 1., 1.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.]])
    return state


def initialize_state_many_objects():
    spacing = np.linspace(2., box_size - 2., 5)
    n_objects = len(spacing)**3
    x1, x2, x3 = np.meshgrid(*([spacing]*3), indexing='ij')
    key = jax.random.PRNGKey(0)
    perturb = jax.random.uniform(key, (dim, n_objects), np.float32, -0.5, 0.5)
    xx = np.concatenate([x1.reshape(1, -1), x2.reshape(1, -1), x3.reshape(1, -1)], axis=0) + perturb 
    q0 = np.ones((1, n_objects))
    state = np.concatenate([xx, q0, np.zeros((9, n_objects))], axis=0)
    return state


def drop_a_stone_3d():
    start_time = time.time()

    object_name = 'sphere'
    # object_name = 'trapezoid'

    cube_func = lambda x: np.max(np.absolute(x), axis=-1) - 1.

    # object_3D = generate_template_object(object_name, 20)
    # object_3D.morph_into_shape(cube_func)

    object_3D = generate_template_object(object_name, 6)

    directions = object_3D.get_directions()
    connectivity = object_3D.get_connectivity()
    vertices = object_3D.get_vertices()
    params = np.ones(len(vertices))
    output_vtk_3D_shape(vertices, connectivity, f"data/vtk/3d/vedo/{object_name}.vtk")

    state = initialize_state_many_objects()
    polyhedra_inertias_no_rotation, polyhedron_vol, ref_centroid = compute_inertia_tensors(params, directions, connectivity, state[3:7].T)

    num_steps = 5000
    dt = 5*1e-4
    states = [state]
    energy = []
    for i in range(num_steps):
        rhs_func = lambda variable: state_rhs_func(params, directions, connectivity, variable) 
        state = runge_kutta_4(state, rhs_func, dt)
        if i % 20 == 0:
            e = compute_energy(params, directions, connectivity, state)
            print(f"\nstep {i}, total energy={e}, quaternion square sum: {np.sum(state[3:7]**2)}")
            # print(f"state=\n{state}")
            if np.any(np.isnan(state)):
                print(f"state=\n{state}")
                break
            energy.append(e)
            states.append(state)

    states = np.array(states)
    energy = np.array(energy)

    end_time = time.time()
    print(f"Time elapsed {end_time-start_time}")
    print(f"Platform: {xla_bridge.get_backend().platform}")

    np.save('data/numpy/vedo/ref_centroid.npy', ref_centroid)
    np.save('data/numpy/vedo/states.npy', states)

    plot_energy(energy, 'data/pdf/energy3d.pdf')
    vedo_plot(object_name, ref_centroid, states)


# def test_vis():
#     object_name = 'sphere'

#     cube_func = lambda x: np.max(np.absolute(x), axis=-1) - 1.

#     object_3D = generate_template_object(object_name, 6)
#     object_3D.morph_into_shape(cube_func)

#     # object_3D = generate_template_object(object_name, 10)

#     directions = object_3D.get_directions()
#     connectivity = object_3D.get_connectivity()
#     vertices = object_3D.get_vertices()
#     params = np.ones(len(vertices))
#     output_vtk_3D_shape(vertices, connectivity, f"data/vtk/3d/vedo/{object_name}.vtk")
#     state = initialize_state_3_objects()
#     _, _, ref_centroid = compute_inertia_tensors(params, directions, connectivity, state[3:7].T)

#     states = []
#     num_steps = 10
#     for i in range(num_steps):
#         state = np.array([[10., 10., 10.],
#                           [10., 10., 10.],
#                           [2. * (1 - i/num_steps), 6., 10.],
#                           [1., 1., 1.],
#                           [0., 0., 0.],
#                           [0., 0., 0.],
#                           [0., 0., 0.],
#                           [0., 0., 0.],
#                           [0., 0., 0.],
#                           [0., 0., 0.],
#                           [0., 0., 0.],
#                           [0., 0., 0.],
#                           [0., 0., 0.]])
#         states.append(state)

#     vedo_plot(object_name, ref_centroid, np.array(states))

if __name__ == '__main__':
    # drop_a_stone_3d()
    # test_vis()
    vedo_plot('sphere')
