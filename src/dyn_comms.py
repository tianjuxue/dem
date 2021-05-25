import jax
import jax.numpy as np


def wall_eval_sdf(wall_loc, direction, axis, point):
    return np.where(direction, point[axis] - wall_loc, wall_loc - point[axis])

wall_grad_sdf = jax.grad(wall_eval_sdf, argnums=-1)
batch_wall_eval_sdf = jax.jit(jax.vmap(wall_eval_sdf, in_axes=(None, None, None, 0), out_axes=0))
batch_wall_grad_sdf = jax.jit(jax.vmap(wall_grad_sdf, in_axes=(None, None, None, 0), out_axes=0))


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