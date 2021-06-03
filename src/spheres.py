import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge
import time
import matplotlib.pyplot as plt
from .dyn_comms import runge_kutta_4
from .arguments import args
from .shape3d import quats_mul


dim = args.dim
gravity = args.gravity


def plot_energy(energy):
    plt.figure(num=10, figsize=(6, 6))
    plt.plot(20*np.arange(1, len(energy) + 1, 1), energy, marker='o',  markersize=2, linestyle="-", linewidth=1, color='blue')
    plt.xlabel("Time steps")
    plt.ylabel("Energy")
    plt.savefig('data/pdf/energy_spheres.pdf')


def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = np.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)


def compute_sphere_inertia_tensors(radius, n_objects):
    vol = 4./3.*np.pi*radius**3
    inertias = np.array([2./5.*vol*radius**2 * np.eye(dim)] * n_objects)
    return inertias, vol


@jax.jit
def compute_energy(radius, state):
    x = state[0:3]
    q = state[3:7]
    v = state[7:10]
    w = state[10:13]
    inertias, vol = compute_sphere_inertia_tensors(radius, state.shape[1])
    total_energy = 1./2. * np.sum(w.T * np.squeeze(inertias @ np.expand_dims(w.T, axis=-1))) + 1./2. * vol * np.sum(v**2) + vol * gravity * np.sum(x[2])
    return total_energy



@jax.jit
def state_rhs_func(radius, state):
    n_objects = state.shape[1]
    x = state[0:3].T
    q = state[3:7].T
    v = state[7:10].T
    w = state[10:13].T

    inertias, vol = compute_sphere_inertia_tensors(radius, n_objects)

    mutual_vectors = x[None, :, :] - x[:, None, :]
    mutual_distances = np.sqrt(np.sum(mutual_vectors**2, axis=-1))
    middle_points = (x[:, None, :] + x[:, None, :]) / 2.

    mutual_distances = fill_diagonal(mutual_distances, 1.)

    mutual_unit_vectors = mutual_vectors / mutual_distances[:, :, None]
    forces = -1e5 * np.where(mutual_distances < 2 * radius, 2 * radius - mutual_distances, 0.)[..., None] * mutual_unit_vectors
    arms = middle_points - x[:, None, :]
    torques = np.cross(arms, forces)
    mutual_reactions = np.concatenate((np.sum(forces, axis=1), np.sum(torques, axis=1)), axis=-1)

    I_inv = np.linalg.inv(inertias) 
 
    x_bottom = jax.ops.index_add(x, jax.ops.index[:, 2], -radius)
    wall_forces = -1e5 * np.where(x_bottom[:, 2] < 0., x_bottom[:, 2] , 0.)[:, None] * np.array([[0., 0., 1.]])
    wall_arms = x_bottom - x
    wall_torques = np.cross(wall_arms, wall_forces)
    wall_reactions = np.concatenate((wall_forces, wall_torques), axis=-1)

    contact_reactions = mutual_reactions + wall_reactions

    dx_rhs = v.T

    w_quat = np.concatenate([np.zeros((1, n_objects)), w.T], axis=0)
    dq_rhs = 0.5 * quats_mul(w_quat.T, q).T

    contact_forces = contact_reactions[:, :3]
    dv_rhs = (contact_forces / vol + np.array([[0., 0., -gravity]])).T

    contact_torques = contact_reactions[:, 3:]
    M = np.expand_dims(contact_torques, axis=-1)
    wIw = np.expand_dims(np.cross(w, np.squeeze(inertias @  np.expand_dims(w, axis=-1))), axis=-1)

    # Check the correctness of using this reshape
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
    radius = 1.
    state = initialize_state_3_objects()
    num_steps = 3000
    dt = 5*1e-4
    states = [state]
    energy = []
    for i in range(num_steps):
        rhs_func = lambda variable: state_rhs_func(radius, variable) 
        state = runge_kutta_4(state, rhs_func, dt)
        if i % 20 == 0:
            e = compute_energy(radius, state)
            print(f"\nstep {i}, total energy={e}, quaternion square sum: {np.sum(state[3:7]**2)}")
            print(f"state=\n{state}")
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

    plot_energy(energy)


if __name__ == '__main__':
    drop_a_stone_3d()
