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
batch_eval_sdf, batch_grad_sdf
from .io import output_vtk_3D_shape
from .dyn_comms import batch_wall_eval_sdf, batch_wall_grad_sdf, get_frictionless_force, runge_kutta_4


gravity = args.gravity
box_size = 20.


def get_reaction(phy_seeds, x, forces):
    toque_arms = phy_seeds - x.reshape(1, -1)
    torques = np.cross(toque_arms, forces)
    f = np.sum(forces, axis=0)
    t = np.sum(torques, axis=0)
    return np.concatenate([f, t])


def compute_mutual_reaction(params, directions, connectivity, ref_centroid, x_o1, q_o1, index1, x_o2, q_o2, index2):
    '''Force (f1, f2) and torque (t) by object 2 on object 1
    '''
    def f3(_):
        # object 2 is master (use seeds of object 2), object 1 is slave
        phy_seeds_o2 = get_phy_seeds(params, directions, connectivity, ref_centroid, x_o2, q_o2)
        level_set_func = partial(batch_eval_sdf, params, directions, connectivity, ref_centroid, x_o1, q_o1)
        level_set_grad = partial(batch_grad_sdf, params, directions, connectivity, ref_centroid, x_o1, q_o1)
        forces = -get_frictionless_force(phy_seeds_o2, level_set_func, level_set_grad)
        reaction = get_reaction(phy_seeds_o2, x_o1, forces)
        return reaction

    def f2(_):
        # object 1 is master (use seeds of object 1), object 2 is slave
        phy_seeds_o1 = get_phy_seeds(params, directions, connectivity, ref_centroid, x_o1, q_o1)
        level_set_func = partial(batch_eval_sdf, params, directions, connectivity, ref_centroid, x_o2, q_o2)
        level_set_grad = partial(batch_grad_sdf, params, directions, connectivity, ref_centroid, x_o2, q_o2)
        forces = get_frictionless_force(phy_seeds_o1, level_set_func, level_set_grad)
        reaction = get_reaction(phy_seeds_o1, x_o1, forces)
        return jax.lax.cond(index1 < index2, lambda _: reaction, f3, _)

    def f1(_):
        # If the origin distance between o1 and o2 is larger than twice the max radius, no mutual force exists.
        max_radius = np.max(params)
        phy_origin_o1 = reference_to_physical(x_o1, q_o1, ref_centroid, np.array([0., 0., 0.]))
        phy_origin_o2 = reference_to_physical(x_o2, q_o2, ref_centroid, np.array([0., 0., 0.]))
        mutual_distance = np.sqrt(np.sum((phy_origin_o1 - phy_origin_o2)**2))
        return jax.lax.cond(mutual_distance > 2 * max_radius, lambda _: np.zeros(6), f2, _)

    # Do not compute anything if o1 and o2 are the same object.
    return jax.lax.cond(index1==index2, lambda _: np.zeros(6), f1, None) 


batch_compute_mutual_reaction_tmp = jax.vmap(compute_mutual_reaction, in_axes=(None,)*7 + (0,)*3, out_axes=0)
batch_compute_mutual_reaction = jax.vmap(batch_compute_mutual_reaction_tmp, in_axes=(None,)*4 + (0,)*3 + (None,)*3, out_axes=0)


def compute_wall_reaction(params, directions, connectivity, ref_centroid, x, q):
    phy_seeds = get_phy_seeds(params, directions, connectivity, ref_centroid, x, q)

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


batch_compute_wall_reaction = jax.vmap(compute_wall_reaction, in_axes=(None, None, None, None, 0, 0), out_axes=0)


@jax.jit
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
 
    n_objects = state.shape[1]
    x = state[0:3]
    q = state[3:7]
    v = state[7:10]
    w = state[10:13]
    polyhedra_intertias, polyhedron_vol, ref_centroid = compute_inertia_tensors(params, directions, connectivity, q.T)
    I_inv = np.linalg.inv(polyhedra_intertias) 
    paired_reactions = batch_compute_mutual_reaction(params, directions, connectivity, ref_centroid, x.T, q.T, np.arange(n_objects), x.T, q.T, np.arange(n_objects))
    mutual_reactions = np.sum(paired_reactions, axis=1)
    wall_reactions = batch_compute_wall_reaction(params, directions, connectivity, ref_centroid, x.T, q.T)
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
    return rhs


def vedo_plot(object_name, ref_centroid, states):
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

    for state in states:
        xx = state[0:3]
        qq = state[3:7]

        for i in range(n_objects):
            x = xx.T[i]
            q = qq.T[i]
            r = R.from_quat([q[1], q[2], q[3], q[0]])
            rotvec = r.as_rotvec()
            angle = np.linalg.norm(rotvec)
            vec = rotvec/angle
            stones[i].pos(x).RotateWXYZ(angle * 180.0 / np.pi, vec[0], vec[1], vec[2])

        plotter = vedo.show(world, *stones)

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
                      [2., 6., 10.],
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


def test_vis():
    object_name = 'sphere'

    cube_func = lambda x: np.max(np.absolute(x), axis=-1) - 1.

    object_3D = generate_template_object(object_name, 20)
    object_3D.morph_into_shape(cube_func)

    # object_3D = generate_template_object(object_name, 10)

    directions = object_3D.get_directions()
    connectivity = object_3D.get_connectivity()
    vertices = object_3D.get_vertices()
    params = np.ones(len(vertices))
    output_vtk_3D_shape(vertices, connectivity, f"data/vtk/3d/vedo/{object_name}.vtk")
    state = initialize_state_3_objects()
    _, _, ref_centroid = compute_inertia_tensors(params, directions, connectivity, state[3:7].T)

    states = []
    num_steps = 10
    for i in range(num_steps):
        state = np.array([[10., 10., 10.],
                          [10., 10., 10.],
                          [2. * (1 - i/num_steps), 6., 10.],
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
        states.append(state)

    vedo_plot(object_name, ref_centroid, np.array(states))


def drop_a_stone_3d():
    start_time = time.time()

    object_name = 'sphere'

    object_3D = generate_template_object(object_name, 20)
    object_3D.morph_into_shape(cube_func)

    # object_3D = generate_template_object(object_name, 10)

    directions = object_3D.get_directions()
    connectivity = object_3D.get_connectivity()
    vertices = object_3D.get_vertices()
    params = np.ones(len(vertices))
    output_vtk_3D_shape(vertices, connectivity, f"data/vtk/3d/vedo/{object_name}.vtk")

    state = initialize_state_3_objects()
    _, _, ref_centroid = compute_inertia_tensors(params, directions, connectivity, state[3:7].T)

    num_steps = 100
    dt = 5*1e-4
    states = [state]
    energy = []
    for i in range(num_steps):
        rhs_func = lambda variable: state_rhs_func(params, directions, connectivity, variable)
        state = runge_kutta_4(state, rhs_func, dt)
        e = compute_energy(params, directions, connectivity, state)
        if i % 20 == 0:
            print(f"\nstep {i}, total energy={e}, state=\n{state}")
            print(f"quaternion square sum: {np.sum(state[3:7]**2)}")
            energy.append(e)
            states.append(state)

    end_time = time.time()
    print(f"Time elapsed {end_time-start_time}")
    print(f"Platform: {xla_bridge.get_backend().platform}")

    plot_energy(np.array(energy))
    vedo_plot(object_name, ref_centroid, np.array(states))


if __name__ == '__main__':
    # drop_a_stone_3d()
    test_vis()
