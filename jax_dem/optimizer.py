import jax
import jax.numpy as np
import numpy as onp
from jax import linear_util as lu
from jax.flatten_util import ravel_pytree
from functools import partial
from jax.tree_util import tree_map
from jax_dem.io import vedo_plot
import scipy.optimize as opt


def ravel_first_arg(f, unravel):
    return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped


@lu.transformation
def ravel_first_arg_(unravel, y_flat, *diff_args):
    y = unravel(y_flat)
    ans = yield (y,) + diff_args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


# @partial(jax.jit, static_argnums=(0,))
def odeint(func, y0, ts, *diff_args):
    y0, unravel = ravel_pytree(y0)
    func = ravel_first_arg(func, unravel)
    out = rk4(func, y0, ts, *diff_args)
    return jax.vmap(unravel)(out)


def rk4(f, y0, ts, *diff_args):
    def step(state, t_crt):
        y_prev, t_prev = state
        h = t_crt - t_prev
        k1 = h * f(y_prev, t_prev, *diff_args)
        k2 = h * f(y_prev + k1/2., t_prev + h/2., *diff_args)
        k3 = h * f(y_prev + k2/2., t_prev + h/2., *diff_args)
        k4 = h * f(y_prev + k3, t_crt + h, *diff_args)
        y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t_crt), y

    # _, ys = jax.lax.scan(step, (y0, ts[0]), ts[1:])

    ys = []
    energy = []
    state = (y0, ts[0])
    for (i, t_crt) in enumerate(ts[1:]):
        state, y = step(state, t_crt)
        # ys.append(y)
        if i % 20 == 0:
            # e = compute_energy(radii, y)
            # print(f"\nstep {i}, total energy={e}, quaternion square sum: {np.sum(y[3:7]**2)}")
            print(f"step {i}")
            if np.any(np.isnan(state[0])):
                print(f"Found np.nan, so break")                
                break
        ys.append(y)
            # energy.append(e)
    # energy = np.array(energy)
    ys = np.array(ys)

    return ys


def get_aug_rhs_func(state_rhs_func, ys, ts):

    # Assumption: Uniform time step

    def aug_rhs_func(aug, t, *diff_args):
        y, y_bar, diff_args_bar = aug
        y_dot, vjpfun = jax.vjp(lambda y, *diff_args: state_rhs_func(y, -t, *diff_args), y, *diff_args)
        return (-y_dot, *vjpfun(y_bar))
    return jax.jit(aug_rhs_func)


def optimize(initials, ts, obj_func, state_rhs_func, bounds=None):
    x_ini, unravel = ravel_pytree(initials)
    aug_rhs_func = get_aug_rhs_func(state_rhs_func)
    def objective(x):
        print(f"Evaluating objective value...")
        y0, diff_args = unravel(x)
        ys = odeint(state_rhs_func, y0, ts, *diff_args)
        obj_val = obj_func(ys[-1])
        objective.aug0 = (ys[-1], jax.grad(obj_func)(ys[-1]), tree_map(np.zeros_like, diff_args))
        objective.diff_args = diff_args
        print(f"y0 = \n{y0}")
        print(f"yf = \n{ys[-1]}")
        # print(f"ys = {ys[:, 2, 0]}")
        print(f"obj_val = {obj_val}")
        return obj_val

    def derivative(x):
        diff_args = objective.diff_args
        aug0 = objective.aug0  
        ys, ys_bar, diff_args_bar = odeint(aug_rhs_func, aug0, -ts[::-1], *diff_args)

        print(f"der = \n{ys_bar[-1]}")
        print(ys[-1])

        der_val, _ = ravel_pytree(tree_map(lambda x: x[-1], [ys_bar, diff_args_bar]))

        exit()

        # print(f"der = {der_val}")

        # der_val, _ = ravel_pytree([ys_bar[-1], diff_args_bar[0][-1]])
        # der_val = jax.ops.index_update(der_val, -1, 0.)
        # der_val = jax.ops.index_update(der_val, -5, 0.)
        
        # 'L-BFGS-B' requires the following conversion, otherwise we get an error message saying
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


def simulate(initials, ts, state_rhs_func):
    y0, diff_args = initials
    ys = odeint(state_rhs_func, y0, ts, *diff_args)

    # plot_energy(energy, f'data/pdf/energy_{object_name}.pdf')

    return ys


if __name__ == '__main__':
    simulate_odeint()
