import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
import argparse
import json
import os
from functools import partial
import matplotlib.pyplot as plt
from .polygon import get_points, args, batch_forward
from .general_utils import show_contours
from . import arguments


def eval_mass(radius_sample):
    pointsA, pointsB = get_points(radius_sample)
    triangle_areas =  1./2 * np.absolute(np.cross(pointsA, pointsB))
    polygon_area = np.sum(triangle_areas)
    triagnle_centroids = 2./3. * 1./2 * (pointsA + pointsB)
    polygon_centroid = np.sum((triangle_areas.reshape(-1, 1) * triagnle_centroids), axis=0) / polygon_area
    triangle_inertias = 1./6. * triangle_areas * (np.sum(pointsA * pointsA, axis=1) +  
        np.sum(pointsA * pointsB, axis=1) + np.sum(pointsB * pointsB, axis=1))
    polygon_inertia = np.sum(triangle_inertias)
    return polygon_area, polygon_inertia, polygon_centroid, pointsA


def ground_level_set(points):
    '''
    Parameter
    ---------
    points: numpy array of shape (batch, dim)
    '''
    heights = points[:, 1]
    return heights
    # return 


def ground_level_set_grad(points):
    '''
    Parameter
    ---------
    points: numpy array of shape (batch, dim)
    '''
    directions = np.vstack([np.zeros(len(points)), np.ones(len(points))]).T
    return directions


def get_frictionless_force(seeds, level_set, level_set_grad):
    stiffness = 1e3
    signed_distances = level_set(seeds)
    directions = level_set_grad(seeds)
    forces = stiffness * np.where(signed_distances < 0., -signed_distances, 0.).reshape(-1, 1) * directions
    return forces


def explicit_Euler(states, rhs_func, dt):
    updated_states = states + dt * rhs_func(states)
    return updated_states


def get_seeds(states, seeds_wrt_centroid_initial):
    x1, x2, theta, v1, v2, omega = states
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    seeds_wrt_centroid = seeds_wrt_centroid_initial @ rot.T
    seeds = seeds_wrt_centroid + np.array([[x1, x2]])
    return seeds


def compute_force(states, area, inertia, seeds_wrt_centroid_initial):
    x1, x2, theta, v1, v2, omega = states
    seeds = get_seeds(states, seeds_wrt_centroid_initial)
    forces = get_frictionless_force(seeds, ground_level_set, ground_level_set_grad)
    torques = np.cross(seeds_wrt_centroid_initial, forces)
    f1, f2 = np.sum(forces, axis=0)
    gravity = -9.8
    f2 += gravity * area
    t = np.sum(torques)
    return np.array([v1, v2, omega, f1/area, f2/area, t/inertia])


def plot_seeds(seeds, fig_no):
    plt.figure(num=fig_no, figsize=(8, 8))
    plt.axis('equal')
    plt.scatter(seeds[:, 0], seeds[:, 1], color='blue', s=2)
    plt.plot([-2, 2], [0, 0], color='black', linewidth=2)

    plt.xlim([-3, 3])
    plt.ylim([-1, 4])


def drop_a_stone_2d():
    params = np.load('data/numpy/training/radius_samples.npy')[1]
    area, inertia, centroid, seeds = eval_mass(params)
    x1_initial = 0.
    x2_initial = 2.
    theta_initial = 0.
    v1_initial = 0.
    v2_initial = 0.
    omega_initial = 0.
    seeds_wrt_centroid_initial = seeds - centroid.reshape(1, -1)
    states_initial = np.array([x1_initial, x2_initial, theta_initial, v1_initial, v2_initial, omega_initial])
    dt = 1e-3
    num_steps = 1000
    rhs_func = partial(compute_force, area=area, inertia=inertia, seeds_wrt_centroid_initial=seeds_wrt_centroid_initial)
    states = states_initial
    for i in range(num_steps):
        # print(compute_force(states, area, inertia, seeds_wrt_centroid_initial))
        states = explicit_Euler(states, rhs_func, dt)
        if i % 50 == 0:
            seeds = get_seeds(states, seeds_wrt_centroid_initial)
            plot_seeds(seeds, i)
        print(states)


def debug():
    # params = np.ones(args.latent_size)
    params = np.load('data/numpy/training/radius_samples.npy')[1]
    print(params.shape)
    polygon_area, polygon_inertia, polygon_centroid, seeds = eval_mass(params)
    print(polygon_area)
    print(polygon_centroid)

    show_contours(batch_forward, params, 0, args)
    plt.scatter(polygon_centroid[0], polygon_centroid[1])


if __name__ == '__main__':
    drop_a_stone_2d()
    # debug()
    plt.show()