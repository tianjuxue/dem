import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge
import argparse
import json
import os
import time
import glob
import shutil
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .polygon import args, get_phy_seeds, batch_get_phy_seeds, batch_eval_sdf, batch_grad_sdf, eval_mass, reference_to_physical
from .general_utils import show_contours
from . import arguments


#TODO: global variable
gravity = 9.8


def bottom_eval_sdf(point):
    return point[1]

bottom_grad_sdf = jax.grad(bottom_eval_sdf)
batch_bottom_eval_sdf = jax.vmap(bottom_eval_sdf, in_axes=0, out_axes=0)
batch_bottom_grad_sdf = jax.vmap(bottom_grad_sdf, in_axes=0, out_axes=0)


def top_eval_sdf(point):
    return 20. - point[1]

top_grad_sdf = jax.grad(top_eval_sdf)
batch_top_eval_sdf = jax.vmap(top_eval_sdf, in_axes=0, out_axes=0)
batch_top_grad_sdf = jax.vmap(top_grad_sdf, in_axes=0, out_axes=0)


def left_eval_sdf(point):
    return point[0]

left_grad_sdf = jax.grad(left_eval_sdf)
batch_left_eval_sdf = jax.vmap(left_eval_sdf, in_axes=0, out_axes=0)
batch_left_grad_sdf = jax.vmap(left_grad_sdf, in_axes=0, out_axes=0)


def right_eval_sdf(point):
    return 20. - point[0]

right_grad_sdf = jax.grad(right_eval_sdf)
batch_right_eval_sdf = jax.vmap(right_eval_sdf, in_axes=0, out_axes=0)
batch_right_grad_sdf = jax.vmap(right_grad_sdf, in_axes=0, out_axes=0)


def get_frictionless_force(phy_seeds, level_set_func, level_set_grad):
    stiffness = 1e5
    signed_distances = level_set_func(phy_seeds)
    directions = level_set_grad(phy_seeds)
    forces = stiffness * np.where(signed_distances < 0., -signed_distances, 0.).reshape(-1, 1) * directions   
    return forces


def explicit_euler(variable, rhs, dt):
    return variable + dt * rhs(variable)


def runge_kutta_4(variable, rhs, dt):
    y_0 = variable
    k_0 = rhs(y_0)
    k_1 = rhs(y_0 + dt/2 * k_0)
    k_2 = rhs(y_0 + dt/2 * k_1)
    k_3 = rhs(y_0 + dt * k_2)
    k = 1./6. * (k_0 + 2. * k_1 + 2. * k_2 + k_3)
    y_1 = y_0 + dt * k
    return y_1


def get_reaction(phy_seeds, x1, x2, forces):
    toque_arms = phy_seeds - np.array([[x1, x2]])
    torques = np.cross(toque_arms, forces)
    f1, f2 = np.sum(forces, axis=0)
    t = np.sum(torques)
    return np.array([f1, f2, t]) 


def compute_mutual_reaction(params, ref_centroid, x1_o1, x2_o1, theta_o1, index1, x1_o2, x2_o2, theta_o2, index2):
    '''Force (f1, f2) and torque (t) by object 2 on object 1
    '''
    def f3(_):
        # object 2 is master (use seeds of object 2), object 1 is slave
        phy_seeds_o2 = get_phy_seeds(params, ref_centroid, x1_o2, x2_o2, theta_o2)
        level_set_func = partial(batch_eval_sdf, params, ref_centroid, x1_o1, x2_o1, theta_o1)
        level_set_grad = partial(batch_grad_sdf, params, ref_centroid, x1_o1, x2_o1, theta_o1)
        forces = -get_frictionless_force(phy_seeds_o2, level_set_func, level_set_grad)
        reaction = get_reaction(phy_seeds_o2, x1_o1, x2_o1, forces)
        return reaction

    def f2(_):
        # object 1 is master (use seeds of object 1), object 2 is slave
        phy_seeds_o1 = get_phy_seeds(params, ref_centroid, x1_o1, x2_o1, theta_o1)
        level_set_func = partial(batch_eval_sdf, params, ref_centroid, x1_o2, x2_o2, theta_o2)
        level_set_grad = partial(batch_grad_sdf, params, ref_centroid, x1_o2, x2_o2, theta_o2)
        forces = get_frictionless_force(phy_seeds_o1, level_set_func, level_set_grad)
        reaction = get_reaction(phy_seeds_o1, x1_o1, x2_o1, forces)
        return jax.lax.cond(index1 < index2, lambda _: reaction, f3, _)

    def f1(_):
        # If the origin distance between o1 and o2 is larger than twice the max radius, no mutual force exists.
        max_radius = np.max(params)
        phy_origin_o1 = reference_to_physical(x1_o1, x2_o1, theta_o1, ref_centroid, np.array([0., 0.]))
        phy_origin_o2 = reference_to_physical(x1_o2, x2_o2, theta_o2, ref_centroid, np.array([0., 0.]))
        mutual_distance = np.sqrt(np.sum((phy_origin_o1 - phy_origin_o2)**2))
        return jax.lax.cond(mutual_distance > 2 * max_radius, lambda _: np.array([0., 0., 0.]), f2, _)

    # Do not compute anything if o1 and o2 are the same object.
    return jax.lax.cond(index1==index2, lambda _: np.array([0., 0., 0.]), f1, None) 


batch_compute_mutual_reaction_tmp = jax.vmap(compute_mutual_reaction, in_axes=(None, None, None, None, None, None, 0, 0, 0, 0), out_axes=0)
batch_compute_mutual_reaction = jax.vmap(batch_compute_mutual_reaction_tmp, in_axes=(None, None, 0, 0, 0, 0, None, None, None, None), out_axes=0)


def compute_wall_reaction(params, ref_centroid, x1, x2, theta):
    phy_seeds = get_phy_seeds(params, ref_centroid, x1, x2, theta)
    forces_bottom = get_frictionless_force(phy_seeds, batch_bottom_eval_sdf, batch_bottom_grad_sdf)
    reaction_bottom = get_reaction(phy_seeds, x1, x2, forces_bottom)
    forces_top = get_frictionless_force(phy_seeds, batch_top_eval_sdf, batch_top_grad_sdf)
    reaction_top = get_reaction(phy_seeds, x1, x2, forces_top)
    forces_left = get_frictionless_force(phy_seeds, batch_left_eval_sdf, batch_left_grad_sdf)
    reaction_left = get_reaction(phy_seeds, x1, x2, forces_left)
    forces_right = get_frictionless_force(phy_seeds, batch_right_eval_sdf, batch_right_grad_sdf)
    reaction_right = get_reaction(phy_seeds, x1, x2, forces_right)
    return reaction_bottom + reaction_top + reaction_left + reaction_right


batch_compute_wall_reaction = jax.vmap(compute_wall_reaction, in_axes=(None, None, 0, 0, 0), out_axes=0)


@jax.jit
def state_rhs_func(params, state):
    '''
    Parameter
    ---------
    params: numpy array of shape (n_params,)
    state: numpy array of shape (6, n_objects)

    Returns
    -------
    rhs: numpy array of shape (6, n_objects)
    '''
    area, inertia, ref_centroid = eval_mass(params)
    x1, x2, theta, v1, v2, omega = state
    n_objects = state.shape[1]
    paired_reactions = batch_compute_mutual_reaction(params, ref_centroid, x1, x2, theta, np.arange(n_objects), x1, x2, theta, np.arange(n_objects))
    mutual_reactions = np.sum(paired_reactions, axis=1)
    wall_reactions = batch_compute_wall_reaction(params, ref_centroid, x1, x2, theta)
    contact_reactions = (mutual_reactions + wall_reactions) / np.array([[area, area, inertia]])
    reactions = contact_reactions + np.array([[0., -gravity, 0.]])
    rhs = np.concatenate([v1.reshape(1, -1), v2.reshape(1, -1), omega.reshape(1, -1), reactions.T], axis=0)
    return rhs

jac_rhs_params = jax.jacrev(state_rhs_func, argnums=(0))
jac_rhs_state = jax.jacrev(state_rhs_func, argnums=(1))
batch_jac_rhs_params = jax.jit(jax.vmap(jac_rhs_params, in_axes=(None, 0), out_axes=0))


def plot_seeds(seeds, fig_no):
    plt.figure(num=fig_no, figsize=(12, 12))
    plt.axis('scaled')
    plt.scatter(seeds[:, 0], seeds[:, 1], color='blue', s=2)
    plt.plot([-2, 2], [0, 0], color='black', linewidth=2)
    plt.xlim([-3, 3])
    plt.ylim([-1, 10])


def plot_animation(seeds_collect, step=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    xdata, ydata = [], []
    plt.plot([0, 20], [0, 0], color='red', linewidth=1)
    plt.plot([0, 20], [20, 20], color='red', linewidth=1)
    plt.plot([0, 0], [0, 20], color='red', linewidth=1)
    plt.plot([20, 20], [0, 20], color='red', linewidth=1)

    n_objects = len(seeds_collect[0])

    lines = [plt.plot([], [], marker='o',  markersize=2, linestyle="-", linewidth=1, color='blue')[0] for _ in range(n_objects)]
    # ln, = plt.plot([], [], marker='o',  markersize=3, linestyle="-", linewidth=1, color='blue')

    def init():
        ax.axis('scaled')
        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 21)
        return lines

    def update(i):
        seeds_objects = seeds_collect[i]
        for i in range(n_objects):
            seeds = seeds_objects[i]
            seeds = np.concatenate([seeds, seeds[:1, :]], axis=0)
            lines[i].set_data(seeds[:, 0], seeds[:, 1])
        return lines

    anim = FuncAnimation(fig, update, frames=len(seeds_collect), init_func=init, blit=True)

    if step is None:
        anim.save(f'data/mp4/test.mp4', fps=30, dpi=300)
    else:
        if step == 0:
            files = glob.glob(f'data/mp4/opt/*')
            for f in files:
                os.remove(f)
            # shutil.rmtree(f'data/mp4/opt/', ignore_errors=True)
        anim.save(f'data/mp4/opt/test{step}.mp4', fps=30, dpi=300)

    # plt.show()


def compute_energy(state, area, inertia):
    x1, x2, theta, v1, v2, omega = state
    kinetic_energy = 1./2. * area * (v1**2 + v2**2) + 1./2. * inertia * omega**2
    potential_energy = area * gravity * x2
    total_energy = kinetic_energy + potential_energy
    return total_energy


def plot_energy(energy):
    plt.figure(num=10, figsize=(6, 6))
    plt.plot(20*np.arange(1, len(energy) + 1, 1), energy, marker='o',  markersize=2, linestyle="-", linewidth=1, color='blue')
    plt.xlabel("Time steps")
    plt.ylabel("Energy")
    plt.savefig('data/pdf/energy.pdf')


def initialize_state_1_object():
    state = np.array([[10.],
                       [2.],
                       [0.],
                       [0.],
                       [0.],
                       [0.]])
    return state


def initialize_state_3_objects():
    state = np.array([[10., 10., 10.],
                       [2., 6., 10.],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.]])
    return state


def initialize_state_25_objects():
    spacing = np.linspace(2., 18., 5)
    n_objects = len(spacing)**2
    x1, x2 = np.meshgrid(spacing, spacing, indexing='ij')
    key = jax.random.PRNGKey(0)
    theta = jax.random.uniform(key, (1, n_objects), np.float32, 0., 2*np.pi)
    perturb = jax.random.uniform(key, (2, n_objects), np.float32, -0.5, 0.5)
    xx = np.concatenate([x1.reshape(1, -1), x2.reshape(1, -1)], axis=0) + perturb
    state = np.concatenate([xx, theta, np.zeros((3, n_objects))], axis=0)
    return state


def solve_states(params, num_steps, dt):
    state = initialize_state_1_object()
    states = [state]
    for i in range(num_steps):
        rhs_func = lambda variable: state_rhs_func(params, variable)
        state = runge_kutta_4(state, rhs_func, dt)
        states.append(state)
        # if i % 20 == 0:
        #     print(f"States ODE \nstep {i}")
    return np.array(states)


def drop_a_stone_2d():
    start_time = time.time()
    params = np.load('data/numpy/training/radius_samples.npy')[1]
    area, inertia, ref_centroid = eval_mass(params)
    state = initialize_state_1_object()
    num_steps = 1500
    dt = 5*1e-4
    seeds_collect = []
    energy = []
    for i in range(num_steps):
        rhs_func = lambda variable: state_rhs_func(params, variable)
        state = runge_kutta_4(state, rhs_func, dt)
        x1, x2, theta, v1, v2, omega = state
        e = compute_energy(state, area, inertia)
        if i % 20 == 0:
            phy_seeds = batch_get_phy_seeds(params, ref_centroid, x1, x2, theta)
            seeds_collect.append(phy_seeds)
            print(f"\nstep {i}, \nenergy={e}, total energy={np.sum(e)}")
            energy.append(e)

    end_time = time.time()
    print(f"Time elapsed {end_time-start_time}")
    print(f"Platform: {xla_bridge.get_backend().platform}")

    plot_energy(np.sum(np.array(energy), axis=-1))
    plot_animation(seeds_collect)


if __name__ == '__main__':
    drop_a_stone_2d()
    # debug()
    # plot_animation()