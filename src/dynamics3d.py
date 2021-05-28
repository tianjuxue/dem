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
batch_eval_sdf, batch_grad_sdf, get_rot_mats, batch_reference_to_physical, quat_mul, get_ref_vertices_oriented, rotate_point, get_ref_seeds
from .io import output_vtk_3D_shape
from .dyn_comms import batch_wall_eval_sdf, batch_wall_grad_sdf, get_frictionless_force, runge_kutta_4
from memory_profiler import profile


dim = args.dim
gravity = args.gravity
box_size = 20.


@jax.jit
def get_reaction(phy_seeds, x, forces):
    toque_arms = phy_seeds - x.reshape(1, -1)
    torques = np.cross(toque_arms, forces)
    f = np.sum(forces, axis=0)
    t = np.sum(torques, axis=0)
    return np.concatenate([f, t])


def compute_wall_reaction(params, directions, connectivity, ref_centroid, x, q, phy_seeds):
    # phy_seeds = get_phy_seeds(params, directions, connectivity, ref_centroid, x, q)

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


def compute_mutual_reaction(params, directions, connectivity, ref_centroid, x, q, batch_phy_seeds, index1, index2):
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
        # phy_seeds_o2 = get_phy_seeds(params, directions, connectivity, ref_centroid, x_o2, q_o2)
        level_set_func = partial(batch_eval_sdf, params, directions, connectivity, ref_centroid, x_o1, q_o1)
        level_set_grad = partial(batch_grad_sdf, params, directions, connectivity, ref_centroid, x_o1, q_o1)



        stiffness = 1e5
        signed_distances = level_set_func(phy_seeds_o2)
        direcs = level_set_grad(phy_seeds_o2)
        forces = -stiffness * np.where(signed_distances < 0., -signed_distances, 0.).reshape(-1, 1) * direcs 


        # forces = -get_frictionless_force(phy_seeds_o2, level_set_func, level_set_grad)


        reaction = get_reaction(phy_seeds_o2, x_o1, forces)
        return reaction, np.concatenate([direcs, signed_distances.reshape(-1, 1)], axis=-1)

    def f2(_):
        # object 1 is master (use seeds of object 1), object 2 is slave
        # phy_seeds_o1 = get_phy_seeds(params, directions, connectivity, ref_centroid, x_o1, q_o1)
        level_set_func = partial(batch_eval_sdf, params, directions, connectivity, ref_centroid, x_o2, q_o2)
        level_set_grad = partial(batch_grad_sdf, params, directions, connectivity, ref_centroid, x_o2, q_o2)


        stiffness = 1e5
        signed_distances = level_set_func(phy_seeds_o1)
        direcs = level_set_grad(phy_seeds_o1)
        forces = stiffness * np.where(signed_distances < 0., -signed_distances, 0.).reshape(-1, 1) * direcs 


        # forces = get_frictionless_force(phy_seeds_o1, level_set_func, level_set_grad)
        


        reaction = get_reaction(phy_seeds_o1, x_o1, forces)
        return reaction, np.concatenate([direcs, signed_distances.reshape(-1, 1)], axis=-1)

    return jax.lax.cond(index1 < index2, f2, f3, None)

batch_compute_mutual_reaction_sdf = jax.jit(jax.vmap(compute_mutual_reaction, in_axes=(None,)*7 + (0,)*2, out_axes=(0, 0)))



def compute_mutual_reaction(params, directions, connectivity, ref_vertice_normals, ref_centroid, x, q, batch_phy_seeds, index1, index2):
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
        inds = np.unravel_index(vertices_distances.argmin(), vertices_distances.shape)
        phy_seed_o2 = np.take(phy_seeds_o2, inds[0], axis=0)
        phy_seed_o1 = np.take(phy_seeds_o1, inds[1], axis=0)
        phy_vertices_normal_o2 = np.take(phy_vertices_normals_o2, inds[0], axis=0)
        d = np.dot(phy_vertices_normal_o2, (phy_seed_o2 - phy_seed_o1))
        forces = stiffness * np.where(d > 0, d, 0.) * phy_vertices_normal_o2.reshape(1, -1)
        reaction = get_reaction(phy_seed_o2.reshape(1, -1), x_o1, forces)
        return reaction

    def f2(_):
        # object 1 is master (use seeds of object 1), object 2 is slave
        phy_vertices_normals_o1 = rotate_point(q_o1, ref_vertice_normals)
        vertices_distances = get_mutual_distances(phy_seeds_o1, phy_seeds_o2)
        inds = np.unravel_index(vertices_distances.argmin(), vertices_distances.shape)
        phy_seed_o1 = np.take(phy_seeds_o1, inds[0], axis=0)
        phy_seed_o2 = np.take(phy_seeds_o2, inds[1], axis=0)
        phy_vertices_normal_o1 = np.take(phy_vertices_normals_o1, inds[0], axis=0)
        d = np.dot(phy_vertices_normal_o1, (phy_seed_o1 - phy_seed_o2))
        forces = -stiffness * np.where(d > 0, d, 0.) * phy_vertices_normal_o1.reshape(1, -1)
        reaction = get_reaction(phy_seed_o1.reshape(1, -1), x_o1, forces)

        # print(forces.shape)
        # print(phy_seed_o1)
        # print(phy_seed_o2)
        # print(f"phy_vertices_normal_o1={phy_vertices_normal_o1}")
        # print(f"norm={np.linalg.norm(phy_vertices_normal_o1)}")
        # print(f"d={d}")
        # print(forces)

        return reaction

    # return f2(None)
    return jax.lax.cond(index1 < index2, f2, f3, None)

batch_compute_mutual_reaction = jax.jit(jax.vmap(compute_mutual_reaction, in_axes=(None,)*8 + (0,)*2, out_axes=0))



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


def add_to_target(index, target_index, reaction, target):
    return jax.ops.index_add(target, target_index, reaction)

reduce_at = jax.jit(jax.vmap(add_to_target, in_axes=(0, 0, 0, None), out_axes=0))


def get_mutual_distances(pointsA, pointsB):
    return np.sqrt(np.sum(pointsA**2, axis=1).reshape(-1, 1) + \
        np.sum(pointsB**2, axis=1).reshape(1, -1) -  2 * np.dot(pointsA, pointsB.T))


def state_rhs_func(params, directions, connectivity, state):
    '''
    Parameter
    ---------
    params: numpy array of shape (n_params,)
    directions: numpy array with shape (num_vertices, dim)
    connectivity: numpy array with shape (num_cells, dim)
    state: numpy array of shape (6, n_objects)

    Returns
    -------
    rhs: numpy array of shape (13, n_objects)
    '''

    ref_vertice_normals = compute_normals(params, directions, connectivity)
    n_objects = state.shape[1]
    x = state[0:3]
    q = state[3:7]
    v = state[7:10]
    w = state[10:13]
    polyhedra_intertias, polyhedron_vol, ref_centroid = compute_inertia_tensors(params, directions, connectivity, q.T)
    I_inv = np.linalg.inv(polyhedra_intertias) 

    # Check if two objects are far apart. If so, avoid computing mutual reactions.
    max_radius = np.max(params)
    phy_origins = batch_reference_to_physical(x.T, q.T, ref_centroid, np.array([0., 0., 0.]))
    mutual_distances = get_mutual_distances(phy_origins, phy_origins)
    collision_indices = np.array(np.where(mutual_distances < 2 * max_radius)).T
    collision_indices = collision_indices[np.where(collision_indices[:, 0] != collision_indices[:, 1])]

    ref_seeds = get_ref_seeds(params, directions, connectivity)
    batch_phy_seeds = batch_reference_to_physical(x.T, q.T, ref_centroid, ref_seeds)

    # compute_mutual_reaction(params, directions, connectivity, ref_vertice_normals, 
    #     ref_centroid, x.T, q.T, batch_phy_seeds, 0, 1)
    # exit()

    break1 = time.time()

    reactions, direcs = batch_compute_mutual_reaction_sdf(params, directions, connectivity, 
        ref_centroid, x.T, q.T, batch_phy_seeds, collision_indices[:, 0], collision_indices[:, 1])

    # reactions = batch_compute_mutual_reaction(params, directions, connectivity, ref_vertice_normals, 
    #     ref_centroid, x.T, q.T, batch_phy_seeds, collision_indices[:, 0], collision_indices[:, 1])

    break2 = time.time()
 
    mutual_reactions = np.zeros((n_objects, 6))
    mutual_reactions = np.sum(reduce_at(np.arange(len(collision_indices)), collision_indices[:, 0], reactions, mutual_reactions), axis=0)

    # if len(collision_indices) > 0:
    #     print(f"Dealing with contacts...")
        # print(mutual_reactions)

    wall_reactions = batch_compute_wall_reaction(params, directions, connectivity, ref_centroid, x.T, q.T, batch_phy_seeds)
    contact_reactions = mutual_reactions + wall_reactions

    dx_rhs = v

    w_quat = np.concatenate([np.zeros((1, n_objects)), w], axis=0)
    dq_rhs = 0.5 * quats_mul(w_quat.T, q.T).T

    contact_forces = contact_reactions[:, :3]
    dv_rhs = (contact_forces / polyhedron_vol + np.array([[0., 0., -gravity]])).T

    contact_torques = contact_reactions[:, 3:]
    M = np.expand_dims(contact_torques, axis=-1)
    wIw = np.expand_dims(np.cross(w.T, np.squeeze(polyhedra_intertias @  np.expand_dims(w.T, axis=-1))), axis=-1)
    dw_rhs = (I_inv @ (M - wIw)).reshape(n_objects, 3).T

    rhs = np.concatenate([dx_rhs, dq_rhs, dv_rhs, dw_rhs], axis=0)

    time_elapesed = break2 - break1 
    if time_elapesed > 1:
        print(f"------------------------------------------------------------------Recompile took {time_elapesed}s")
        print(collision_indices)

    if np.any(np.isnan(rhs)):
        print("rhs nan")
        # print(rhs)
        print(collision_indices)
        print(reactions)
        print(direcs)


        np.save('data/numpy/debug/params.npy', params)
        np.save('data/numpy/debug/directions.npy', directions)
        np.save('data/numpy/debug/connectivity.npy', connectivity)
        np.save('data/numpy/debug/ref_centroid.npy', ref_centroid)
        np.save('data/numpy/debug/x.npy', x)
        np.save('data/numpy/debug/q.npy', q)
        np.save('data/numpy/debug/batch_phy_seeds.npy', batch_phy_seeds)
        np.save('data/numpy/debug/collision_indices.npy', collision_indices)

        exit()


    return rhs


def vedo_plot(object_name, ref_centroid=None, states=None):

    if ref_centroid is None:
        ref_centroid = np.load('data/numpy/vedo/ref_centroid.npy')

    if states is None:
        states = np.load('data/numpy/vedo/states.npy')

    vedo.settings.useDepthPeeling = False

    world = vedo.Box([box_size/2., box_size/2., box_size/2.], box_size, box_size, box_size).wireframe()

    stone = vedo.Mesh(f"data/vtk/3d/vedo/{object_name}.vtk").c("red").addShadow(z=0)
    stone.origin(*ref_centroid)

    n_objects = states.shape[-1]
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

        # if plotter.escaped: 
        #     break

    vd.close() 
    # vedo.interactive().close()


def compute_energy(params, directions, connectivity, state):
    x = state[0:3]
    q = state[3:7]
    v = state[7:10]
    w = state[10:13]
    inertias, vol, _ = compute_inertia_tensors(params, directions, connectivity, q.T)
    total_energy = 1./2. * np.sum(w.T * np.squeeze(inertias @ np.expand_dims(w.T, axis=-1))) + 1./2. * vol * np.sum(v**2) + vol * gravity * np.sum(x[2])
    return total_energy
 

def plot_energy(energy):
    plt.figure(num=10, figsize=(6, 6))
    plt.plot(20*np.arange(1, len(energy) + 1, 1), energy, marker='o',  markersize=2, linestyle="-", linewidth=1, color='blue')
    plt.xlabel("Time steps")
    plt.ylabel("Energy")
    plt.savefig('data/pdf/energy3d.pdf')


def initialize_state_1_object():
    state = np.array([10., 10., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(-1, 1)
    return state


def initialize_state_3_objects():
    state = np.array([[10., 10., 10.],
                      [10., 10., 10.],
                      # [2., 6, 10.],
                      [1.1, 3.3, 10.],
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
    spacing = np.linspace(2., 18., 5)
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

    cube_func = lambda x: np.max(np.absolute(x), axis=-1) - 1.

    # object_3D = generate_template_object(object_name, 20)
    # object_3D.morph_into_shape(cube_func)

    object_3D = generate_template_object(object_name, 6)

    directions = object_3D.get_directions()
    connectivity = object_3D.get_connectivity()
    vertices = object_3D.get_vertices()
    params = np.ones(len(vertices))
    output_vtk_3D_shape(vertices, connectivity, f"data/vtk/3d/vedo/{object_name}.vtk")

    state = initialize_state_3_objects()
    polyhedra_intertias_no_rotation, polyhedron_vol, ref_centroid = compute_inertia_tensors(params, directions, connectivity, state[3:7].T)

    num_steps = 10000
    dt = 5*1e-4
    states = [state]
    energy = []
    for i in range(num_steps):
        rhs_func = lambda variable: state_rhs_func(params, directions, connectivity, variable) 
        state = runge_kutta_4(state, rhs_func, dt)
        e = compute_energy(params, directions, connectivity, state)
        if i % 20 == 0:
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

    plot_energy(energy)
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
    drop_a_stone_3d()
    # test_vis()
    # vedo_plot('sphere')
