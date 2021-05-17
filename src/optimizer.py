import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
import argparse
import json
import os
import time
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .polygon import args, get_phy_seeds, batch_get_phy_seeds, batch_eval_sdf, batch_grad_sdf, eval_mass, reference_to_physical
from . import arguments
from .simulator import solve_states, explicit_euler, jac_rhs_state, batch_jac_rhs_params, plot_animation


def objective(params, final_state, initial_mass):
    reg_smooth = np.sum((params - np.roll(params, 1))**2) + \
                 np.sum((params - np.roll(params, 2))**2) + \
                 np.sum((params - np.roll(params, 3))**2)
    area, inertia, ref_centroid = eval_mass(params)
    reg_conserve_mass = (area - initial_mass)**2
    x1, x2, theta, v1, v2, omega = final_state
    # return np.sum(0*reg_smooth + 1e3*reg_conserve_mass + theta)
    return np.sum(-theta)

grad_objective_params = jax.jit(jax.grad(objective, argnums=(0)))
grad_objective_state = jax.jit(jax.grad(objective, argnums=(1)))


@jax.jit
def adjoint_rhs_func(params, state, adjoint):
    jac_f_u = jac_rhs_state(params, state)
    jac_f_u = jac_f_u.reshape(state.size, state.size)
    adjoint = adjoint.reshape(state.size)
    result = -adjoint @ jac_f_u
    return result.reshape(state.shape)


def solve_adjoints(params, states, initial_mass, num_steps, dt):
    adjoint = grad_objective_state(params, states[-1], initial_mass)
    adjoints = [adjoint]
    for i in range(num_steps):
        state = states[num_steps - i]
        rhs_func = lambda variable: adjoint_rhs_func(params, state, variable)
        adjoint = explicit_euler(adjoint, rhs_func, -dt)         
        adjoints.append(adjoint)
        # if i % 20 == 0:
        #     print(f"Adjoint ODE \nstep {i}")

    # print(np.array(adjoints)[-2:])
    # print(np.flip(np.array(adjoints))[:2])
    # print(np.flip(np.array(adjoints), axis=0)[:2])

    return np.flip(np.array(adjoints), axis=0)


def trapezoid_integration(adjoint_1, adjoint_2, jac_f_p_1, jac_f_p_2, dt):
    adjoint_size = adjoint_1.size
    value_1 = adjoint_1.reshape(adjoint_size) @ jac_f_p_1.reshape(adjoint_size, -1)
    value_2 = adjoint_2.reshape(adjoint_size) @ jac_f_p_2.reshape(adjoint_size, -1)
    return (value_1 + value_2) * dt / 2.

batch_trapezoid_integration = jax.jit(jax.vmap(trapezoid_integration, in_axes=(0, 0, 0, 0, None), out_axes=0))


def compute_grad(params, num_steps, dt, initial_mass):
    print(f"\nSolve for states...")
    states = solve_states(params, num_steps, dt)

    J = objective(params, states[-1], initial_mass)
    x1, x2, theta, v1, v2, omega = states[-1]
    print(f"\nJ = {J}")
    print(f"\ntheta = {theta}")
    print(f"\nparams = \n{params}")

    print(f"\nSolve for adjoints...")
    adjoints = solve_adjoints(params, states, initial_mass, num_steps, dt)
 
    print(f"\nCompute pf_pp...")
    batch_jac_f_p = batch_jac_rhs_params(params, states)

    print(f"\nCompute dJ_dp...")
    dJ_dp_tmp1 = np.sum(batch_trapezoid_integration(adjoints[:-1], adjoints[1:], batch_jac_f_p[:-1], batch_jac_f_p[1:], dt), axis=0)

    dJ_dp_tmp1 = 1e-1 * dJ_dp_tmp1 / np.linalg.norm(dJ_dp_tmp1)

    dJ_dp_tmp2 = grad_objective_params(params, states[-1], initial_mass)

    dJ_dp = dJ_dp_tmp1 + dJ_dp_tmp2

    print(f"\ndJ_dp_tmp1 = \n{dJ_dp_tmp1}")
    print(f"\ndJ_dp_tmp2 = \n{dJ_dp_tmp2}")
    print(f"\ndJ_dp = \n{dJ_dp}")
    # dJ_dp = dJ_dp / np.linalg.norm(dJ_dp)
    print(f"\ndJ_dp normalized = \n{dJ_dp}")

    return dJ_dp, states


def output_results(params, states, ref_centroid, num_steps, opt_iter):
    seeds_collect = []
    for i in range(num_steps):
        if i % 20 == 0:
            x1, x2, theta, v1, v2, omega = states[i]
            phy_seeds = batch_get_phy_seeds(params, ref_centroid, x1, x2, theta)
            seeds_collect.append(phy_seeds)

    plot_animation(seeds_collect, opt_iter)


def gradient_descent():
    num_steps = 1500
    dt = 5*1e-4
    params = np.load('data/numpy/training/radius_samples.npy')[1]
    # params = np.ones(len(params))
    initial_mass, inertia, ref_centroid = eval_mass(params)


    for opt_iter in range(100):
        print(f"\n###################################################################################")
        print(f"Gradient descent step {opt_iter}")
        dJ_dp, states = compute_grad(params, num_steps, dt, initial_mass)
        output_results(params, states, ref_centroid, num_steps, opt_iter)
        params = params - dJ_dp


if __name__ == '__main__':
    gradient_descent()
