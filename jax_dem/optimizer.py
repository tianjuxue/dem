import jax
import jax.numpy as np
import numpy as onp
from jax import linear_util as lu
from jax.flatten_util import ravel_pytree
from functools import partial
from jax.tree_util import tree_map
from jax_dem.dynamics import state_rhs_func
from jax_dem.io import vedo_plot
from jax_dem.arguments import args
import scipy.optimize as opt


def ravel_first_arg(f, unravel):
    return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped


@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


# @partial(jax.jit, static_argnums=(0,))
def odeint(func, y0, ts, *args):
    y0, unravel = ravel_pytree(y0)
    func = ravel_first_arg(func, unravel)
    out = rk4(func, y0, ts, *args)
    return jax.vmap(unravel)(out)


def rk4(f, y0, ts, *args):
    def step(state, t_crt):
        y_prev, t_prev = state
        h = t_crt - t_prev
        k1 = h * f(y_prev, t_prev, *args)
        k2 = h * f(y_prev + k1/2., t_prev + h/2., *args)
        k3 = h * f(y_prev + k2/2., t_prev + h/2., *args)
        k4 = h * f(y_prev + k3, t_crt + h, *args)
        y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t_crt), y

    # _, ys = jax.lax.scan(step, (y0, ts[0]), ts[1:])

    ys = []
    state = (y0, ts[0])
    for (i, t_crt) in enumerate(ts[1:]):
        state, y = step(state, t_crt)
        ys.append(y)

        if i % 20 == 0:
            print(f"step {i}")
            if np.any(np.isnan(state[0])):
                print(f"Found np.nan, so break")                
                break
    ys = np.array(ys)

    return ys


@jax.jit
def aug_rhs_func(aug, t, *args):
    y, y_bar, args_bar = aug
    y_dot, vjpfun = jax.vjp(lambda y, *args: state_rhs_func(y, -t, *args), y, *args)
    return (-y_dot, *vjpfun(y_bar))


def optimize(ini_func, obj_func, bounds=None):
    y0, ts, args = ini_func()
    x_ini, unravel = ravel_pytree([y0, args])

    def objective(x):
        print(f"Evaluating objective value...")
        # x = np.array(x)
        y0, args = unravel(x)
        ys = odeint(state_rhs_func, y0, ts, *args)
        obj_val = obj_func(ys[-1])
        objective.initial_carry = (ys[-1], jax.grad(obj_func)(ys[-1]), tree_map(np.zeros_like, args))
        print(f"current x = {x}")
        print(f"obj_val = {obj_val}")
        return obj_val

    def derivative(x):
        initial_carry = objective.initial_carry  
        ys, ys_bar, args_bar = odeint(aug_rhs_func, initial_carry, -ts[::-1], *args)

        der_val, _ = ravel_pytree([ys_bar[-1], args_bar[0][-1]])

        der_val = jax.ops.index_update(der_val, -1, 0.)
        der_val = jax.ops.index_update(der_val, -5, 0.)
        
        # 'L-BFGS-B' requires the following conversion, otherwise you get an error message saying
        # -- input not fortran contiguous -- expected elsize=8 but got 4
        return onp.array(der_val, order='F', dtype=onp.float64)

    options = {'maxiter': 100, 'disp': True}  # CG or L-BFGS-B or Newton-CG
    res = opt.minimize(fun=objective,
                       x0=x_ini,
                       method='L-BFGS-B',
                       jac=derivative,
                       bounds=bounds,
                       callback=None,
                       options=options)


if __name__ == '__main__':
    simulate_odeint()
