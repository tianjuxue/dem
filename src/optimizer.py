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


def objective(params, final_state):
    x1, x2, theta, v1, v2, omega = final_state
    return np.sum(-theta)

grad_objective_state = jax.jit(jax.grad(objective, argnums=(1)))


def regularization(params, initial_mass):
    area, inertia, ref_centroid = eval_mass(params)
    reg_conserve_mass = (area - initial_mass)**2
    return np.sum(reg_conserve_mass)

grad_reg_params = jax.jit(jax.grad(regularization, argnums=(0)))


@jax.jit
def adjoint_rhs_func(params, state, adjoint):
    jac_f_u = jac_rhs_state(params, state)
    jac_f_u = jac_f_u.reshape(state.size, state.size)
    adjoint = adjoint.reshape(state.size)
    result = -adjoint @ jac_f_u
    return result.reshape(state.shape)


def solve_adjoints(params, states, num_steps, dt):
    adjoint = grad_objective_state(params, states[-1])
    adjoints = [adjoint]
    for i in range(num_steps):
        state = states[num_steps - i]
        rhs_func = lambda variable: adjoint_rhs_func(params, state, variable)
        adjoint = explicit_euler(adjoint, rhs_func, -dt)         
        adjoints.append(adjoint)
        # if i % 20 == 0:
        #     print(f"Adjoint ODE \nstep {i}")
    return np.flip(np.array(adjoints), axis=0)


def trapezoid_integration(adjoint_1, adjoint_2, jac_f_p_1, jac_f_p_2, dt):
    adjoint_size = adjoint_1.size
    value_1 = adjoint_1.reshape(adjoint_size) @ jac_f_p_1.reshape(adjoint_size, -1)
    value_2 = adjoint_2.reshape(adjoint_size) @ jac_f_p_2.reshape(adjoint_size, -1)
    return (value_1 + value_2) * dt / 2.

batch_trapezoid_integration = jax.jit(jax.vmap(trapezoid_integration, in_axes=(0, 0, 0, 0, None), out_axes=0))


def smooth_out_gradient(dJ_dp):
    dJ_dp = 1e-1 * dJ_dp / np.max(np.absolute(dJ_dp))
    # dJ_dp = np.where(dJ_dp > 0.05, 0.05, dJ_dp)
    # dJ_dp = np.where(dJ_dp < -0.05, -0.05, dJ_dp)

    effect_range = 5
    coeff = np.exp(-0.25*np.arange(-effect_range + 1, effect_range)**2)
    rolled = np.vstack([np.roll(dJ_dp, i) for i in np.arange(-effect_range + 1, effect_range)]).T
    smoothed = np.sum(coeff * rolled, axis=1)  
    return smoothed


def compute_grad(params, num_steps, dt):
    print(f"\nSolve for states...")
    states = solve_states(params, num_steps, dt)
    J = objective(params, states[-1])
    x1, x2, theta, v1, v2, omega = states[-1]
    print(f"\nJ = {J}")
    print(f"\nparams = \n{params}")

    print(f"\nSolve for adjoints...")
    adjoints = solve_adjoints(params, states, num_steps, dt)
 
    print(f"\nCompute pf_pp...")
    batch_jac_f_p = batch_jac_rhs_params(params, states)

    print(f"\nCompute dJ_dp_obj...")
    dJ_dp_obj = np.sum(batch_trapezoid_integration(adjoints[:-1], adjoints[1:], batch_jac_f_p[:-1], batch_jac_f_p[1:], dt), axis=0)
    print(f"\ndJ_dp = \n{dJ_dp_obj}")
    print(f"min={np.min(dJ_dp_obj)}, max={np.max(dJ_dp_obj)}")

    return dJ_dp_obj, states


def output_results(params, states, num_steps, opt_iter):
    _, _, ref_centroid = eval_mass(params)
    seeds_collect = []
    for i in range(num_steps):
        if i % 20 == 0:
            x1, x2, theta, v1, v2, omega = states[i]
            phy_seeds = batch_get_phy_seeds(params, ref_centroid, x1, x2, theta)
            seeds_collect.append(phy_seeds)

    plot_animation(seeds_collect, opt_iter)


def conserve_mass_gradient_descent(params, initial_mass):
    print("\nRegularizing mass with gradient descent...")
    tol = 1e-5
    res = 1.
    counter = 0
    while res > tol:
        counter += 1
        dJ_dp_reg = grad_reg_params(params, initial_mass)
        params = params - dJ_dp_reg
        res = regularization(params, initial_mass)

    print(f"counter = {counter}")
    return params


def conserve_mass_binary_search(params, initial_mass):
    print("\nRegularizing mass with binary search...")
    small = 0.5
    large = 1.5
    factor = (small + large) / 2.
    tol = 1e-5
    res = 1.
    counter = 0
    while res > tol:
        counter += 1
        mass, _, _ = eval_mass(params * factor)
        if mass > initial_mass:
            large = factor
        else:
            small = factor
        factor = (small + large) / 2.
        res = np.absolute(mass - initial_mass)

    print(f"counter = {counter}")
    return factor * params


def main():
    num_steps = 1500
    dt = 5*1e-4
    params = np.load('data/numpy/training/radius_samples.npy')[1]
    # params = np.ones(len(params))
    initial_mass, _, _ = eval_mass(params)

    for opt_iter in range(100):
        print(f"\n###################################################################################")
        print(f"Gradient descent step {opt_iter}")
        dJ_dp_obj, states = compute_grad(params, num_steps, dt)
        output_results(params, states, num_steps, opt_iter)
        params = params - smooth_out_gradient(dJ_dp_obj)
        params = conserve_mass_gradient_descent(params, initial_mass)


if __name__ == '__main__':
    main()
