import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge
import time
import matplotlib.pyplot as plt
import vedo
from scipy.spatial.transform import Rotation as R
from .dyn_comms import runge_kutta_4
from .arguments import args
from .shape3d import quats_mul, quat_mul, get_rot_mats
from .io import plot_energy


dim = args.dim
gravity = args.gravity
# box_size = args.box_size
box_size = 30.


def env_distance_value(point):
    distances_to_walls = np.concatenate((point - 0., box_size - point))
    return np.min(distances_to_walls)

env_distance_values = jax.vmap(env_distance_value, in_axes=0, out_axes=0)

env_distance_grad = jax.grad(env_distance_value, argnums=0)
env_distance_grads = jax.vmap(env_distance_grad, in_axes=0, out_axes=0)


def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = np.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)


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


def vedo_plot(object_name, radius, states=None):
    if states is None:
        states = np.load(f'data/numpy/vedo/states_{object_name}.npy')
 
    n_objects = states.shape[-1]

    if not hasattr(radius, "__len__"):
        radius = np.array([radius] * n_objects)
    else:
        radius = radius.reshape(-1)

    assert(radius.shape == (n_objects,))

    world = vedo.Box([box_size/2., box_size/2., box_size/2.], box_size, box_size, box_size).wireframe()
    vedo.show(world, axes=4, viewup="z", interactive=0)
    vd = vedo.Video(f"data/mp4/3d/{object_name}.mp4", fps=30)
    # Modify vd.options so that preview on Mac OS is enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    vd.options = "-b:v 8000k -pix_fmt yuv420p"

    for s in range(len(states)):
        x = states[s][0:3].T
        q = states[s][3:7].T
        initial_arrow = radius.reshape(-1, 1) * np.array([[0., 0., 1]])
        rot_matrices = get_rot_mats(q)
        endPoints = np.squeeze(rot_matrices @ initial_arrow[..., None], axis=-1) + x
        arrows = vedo.Arrows(startPoints=x, endPoints=endPoints, c="green")
        balls = vedo.Spheres(centers=x, r=radius, c="red", alpha=0.5)
        plotter = vedo.show(world, balls, arrows, resetcam=False)
        print(f"frame: {s} in {len(states) - 1}")
        vd.addFrame()

    vd.close() 
    # vedo.interactive().close()


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


@jax.jit
def state_rhs_func(radii, state):
    n_objects = state.shape[1]
    x = state[0:3].T
    q = state[3:7].T
    v = state[7:10].T
    w = state[10:13].T
    inertias, vol = compute_sphere_inertia_tensors(radii, n_objects)

    Coulomb_fric_coeff = 0.5
    normal_contact_stiffness = 1e5
    damping_coeff = 1e1
    tangent_fric_coeff = 1e1
    rolling_fric_coeff = 1e-1
 
    mutual_vectors = x[None, :, :] - x[:, None, :]
    mutual_distances,  mutual_unit_vectors = get_unit_vectors(mutual_vectors)
    intersected_distances = radii[None, :] + radii[:, None] - mutual_distances
    elastic_normal_forces = -normal_contact_stiffness * np.where(intersected_distances > 0., intersected_distances, 0.)[:, :, None] * mutual_unit_vectors

    reduced_mass = vol[None, :] * vol[:, None] / (vol[None, :] + vol[:, None])
    contac_points = x[:, None, :] + (radii[:, None] - intersected_distances / 2.)[:, :, None] * mutual_unit_vectors
    arms_self = contac_points - x[:, None, :]
    arms_other = contac_points - x[None, :, :]
    relative_velocity = v[None, :, :] + np.cross(w[None, :, :], arms_other) - v[:, None, :] - np.cross(w[:, None, :], arms_self)
    normal_velocity = np.sum(relative_velocity * mutual_unit_vectors, axis=-1)[:, :, None] * mutual_unit_vectors
    damping_forces = 2 * damping_coeff * np.where(intersected_distances > 0., reduced_mass, 0.)[:, :, None] * normal_velocity

    tangent_velocity = relative_velocity - normal_velocity
    fric_mutual = 2 * tangent_fric_coeff * reduced_mass[:, :, None] * tangent_velocity
    fric_mutual_norms, fric_mutual_unit_vectors = get_unit_vectors(fric_mutual)
    fric_mutual_bounds = Coulomb_fric_coeff * np.sqrt(np.sum(elastic_normal_forces**2, axis=-1))
    fric_mutual_forces = np.where(fric_mutual_norms < fric_mutual_bounds, fric_mutual_norms, fric_mutual_bounds )[:, :, None] * fric_mutual_unit_vectors

    reduced_radius = radii[None, :] * radii[:, None] / (radii[None, :] + radii[:, None])
    relative_w = w[None, :, :] - w[:, None, :] 
    _, relative_w_unit_vectors = get_unit_vectors(relative_w)
    rolling_torque = rolling_fric_coeff * np.linalg.norm(elastic_normal_forces, axis=-1)[:, :, None] * reduced_radius[:, :, None] * relative_w_unit_vectors

    mutual_forces = elastic_normal_forces + damping_forces + fric_mutual_forces
    mutual_torques = np.cross(arms_self, mutual_forces) + rolling_torque
    mutual_reactions = np.concatenate((np.sum(mutual_forces, axis=1), np.sum(mutual_torques, axis=1)), axis=-1)


    env_intersected_distances = radii - env_distance_values(x)
    env_unit_vectors = -env_distance_grads(x)
    env_elastic_normal_forces = -normal_contact_stiffness * np.where(env_intersected_distances > 0., env_intersected_distances, 0.)[:, None] * env_unit_vectors
 
    env_contac_points = x + (radii - env_intersected_distances / 2.)[:, None] * env_unit_vectors
    env_arms = env_contac_points - x
    env_relative_velocity = -v - np.cross(w, env_arms)
    env_normal_velocity = np.sum(env_relative_velocity * env_unit_vectors, axis=-1)[:, None] * env_unit_vectors
    env_damping_forces = 2 * damping_coeff * np.where(env_intersected_distances > 0., vol, 0.)[:, None] * env_normal_velocity

    env_tangent_velocity = env_relative_velocity - env_normal_velocity
    fric_env = 2 * tangent_fric_coeff * vol[:, None] * env_tangent_velocity
    fric_env_norms, fric_env_unit_vectors = get_unit_vectors(fric_env)
    fric_env_bounds = Coulomb_fric_coeff * np.sqrt(np.sum(env_elastic_normal_forces**2, axis=-1))
    fric_env_forces = np.where(fric_env_norms < fric_env_bounds, fric_env_norms, fric_env_bounds)[:, None] * fric_env_unit_vectors

    _, env_relative_w_unit_vectors = get_unit_vectors(-w)
    env_rolling_torque = rolling_fric_coeff * np.linalg.norm(env_elastic_normal_forces, axis=-1)[:, None] * radii[:, None] * env_relative_w_unit_vectors

    env_forces = env_elastic_normal_forces + env_damping_forces + fric_env_forces
    env_torques = np.cross(env_arms, env_forces) + env_rolling_torque
    env_reactions = np.concatenate((env_forces, env_torques), axis=-1)


    contact_reactions = mutual_reactions + env_reactions


    dx_rhs = v.T

    w_quat = np.concatenate([np.zeros((1, n_objects)), w.T], axis=0)
    dq_rhs = 0.5 * quats_mul(w_quat.T, q).T

    contact_forces = contact_reactions[:, :3]
    dv_rhs = (contact_forces / vol[:, None] + np.array([[0., 0., -gravity]])).T

    contact_torques = contact_reactions[:, 3:]
    M = np.expand_dims(contact_torques, axis=-1)
    wIw = np.expand_dims(np.cross(w, np.squeeze(inertias @  np.expand_dims(w, axis=-1))), axis=-1)

    # Check the correctness of using this reshape
    I_inv = np.linalg.inv(inertias) 
    dw_rhs = (I_inv @ (M - wIw)).reshape(n_objects, 3).T

    rhs = np.concatenate([dx_rhs, dq_rhs, dv_rhs, dw_rhs], axis=0)

    return rhs


def initialize_state_3_objects():
    state = np.array([[10., 10.1, 10.],
                      [10., 10.1, 10.],
                      [2., 6, 10.],
                      # [1.1, 3.3, 10.],
                      [1., 1., 1.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [5., 0., 0.],
                      [5., 0., 0.],
                      [-5., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.]])
    return state


def initialize_state_many_objects():
    spacing = np.linspace(2., box_size - 2., 10)
    n_objects = len(spacing)**3
    x1, x2, x3 = np.meshgrid(*([spacing]*3), indexing='ij')
    key = jax.random.PRNGKey(0)
    perturb = jax.random.uniform(key, (dim, n_objects), np.float32, -0.5, 0.5)
    xx = np.concatenate([x1.reshape(1, -1), x2.reshape(1, -1), x3.reshape(1, -1)], axis=0) + perturb 
    q0 = np.ones((1, n_objects))
    state = np.concatenate([xx, q0, np.zeros((9, n_objects))], axis=0)
    return state


def drop_a_stone_3d():
    object_name = 'perfect_ball'
    start_time = time.time()
    state = initialize_state_3_objects()
    radii = np.ones(state.shape[1])
    num_steps = 3000
    dt = 5*1e-4
    states = [state]
    energy = []
    for i in range(num_steps):
        rhs_func = lambda variable: state_rhs_func(radii, variable) 
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

    end_time = time.time()
    print(f"Time elapsed {end_time-start_time}")
    print(f"Platform: {xla_bridge.get_backend().platform}")

    np.save('data/numpy/vedo/states_perfect_ball.npy', states)

    plot_energy(energy, 'data/pdf/energy_perfect_ball.pdf')
    vedo_plot(object_name, radii, states)


if __name__ == '__main__':
    drop_a_stone_3d()
    # vedo_plot('perfect_ball', 1.)
