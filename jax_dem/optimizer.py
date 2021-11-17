import jax
import jax.numpy as np
import numpy as onp
from jax import linear_util as lu
from jax.flatten_util import ravel_pytree
from functools import partial
from jax.tree_util import tree_map
from jax_dem.io import vedo_plot
from jax_dem.utils import get_unit_vectors, quats_mul
import scipy.optimize as opt
import matplotlib.pyplot as plt
from jax_dem.arguments import args

dim = args.dim


def ravel_first_arg(f, unravel):
    return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped


@lu.transformation
def ravel_first_arg_(unravel, y_flat, *diff_args):
    y = unravel(y_flat)
    ans = yield (y,) + diff_args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


# @partial(jax.jit, static_argnums=(0,))
def odeint_ravelled(stepper, f, y0, ts, *diff_args):
    y0, unravel = ravel_pytree(y0)
    f = ravel_first_arg(f, unravel)
    out = odeint(stepper, f, y0, ts, *diff_args)
    return jax.vmap(unravel)(out)


@partial(jax.jit, static_argnums=(2,))
def rk4(state, t_crt, f, *diff_args):
    y_prev, t_prev = state
    h = t_crt - t_prev
    k1 = h * f(y_prev, t_prev, *diff_args)
    k2 = h * f(y_prev + k1/2., t_prev + h/2., *diff_args)
    k3 = h * f(y_prev + k2/2., t_prev + h/2., *diff_args)
    k4 = h * f(y_prev + k3, t_prev + h, *diff_args)
    y_crt = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (y_crt, t_crt), y_crt


@partial(jax.jit, static_argnums=(2,))
def explicit_euler(state, t_crt, f, *diff_args):
    y_prev, t_prev = state
    h = t_crt - t_prev
    y_crt = y_prev + h * f(y_prev, t_prev, *diff_args)
    return (y_crt, t_crt), y_crt


@partial(jax.jit, static_argnums=(2,))
def leapfrog(state, t_crt, f, *diff_args):
    y_prev, t_prev = state
    h = t_crt - t_prev

    # x_prev, q_prev are at time step n
    # v_prev, w_prev are at time step n-1/2
    x_prev = y_prev[:, 0:3]
    q_prev = y_prev[:, 3:7]
    v_prev = y_prev[:, 7:10]
    w_prev = y_prev[:, 10:13]

    rhs = f(y_prev, t_prev, *diff_args)
    rhs_v = rhs[:, 7:10]
    rhs_w = rhs[:, 10:13]

    # v_crt, w_crt are at time step n+1/2
    v_crt = v_prev + h * rhs_v
    w_crt = w_prev + h * rhs_w

    # x_crt, q_crt are at time step n+1
    x_crt = x_prev + h * v_crt
    # Reference: https://doi.org/10.1016/j.powtec.2012.03.023
    # Formula (A.5) in appendix for computing delta_q
    w_crt_norm, w_crt_dir = get_unit_vectors(w_crt)
    delta_q = np.hstack((np.cos(w_crt_norm*h/2)[:, None], w_crt_dir * np.sin(w_crt_norm*h/2)[:, None]))
    q_crt = quats_mul(delta_q, q_prev)

    y_crt = np.hstack((x_crt, q_crt, v_crt, w_crt))
    return (y_crt, t_crt), y_crt


def odeint(stepper, f, y0, ts, *diff_args):

    def stepper_partial(state, t_crt):
        return stepper(state, t_crt, f, *diff_args)

    # _, ys = jax.lax.scan(stepper_partial, (y0, ts[0]), ts[1:])

    ys = []
    state = (y0, ts[0])
    for (i, t_crt) in enumerate(ts[1:]):
        state, y = stepper_partial(state, t_crt)
        if i % 20 == 0:
            print(f"step {i}")
            if not np.all(np.isfinite(y)):
                print(f"Found np.inf or np.nan in y - stop the program") 
                for ind in range(20):
                    print(f"ind =  {ind}")
                    print(ys[ind - 20])      
                exit()
        ys.append(y)
    ys = np.array(ys)
    return ys


def lerp(ts, ys, t):
    assert len(ts) == len(ys), f"ts.shape = {ts.shape}, ys.shape = {ys.shape}"
    # assert t >= ts[0] and t <= ts[-1]

    # Assumption: Uniform time step
    dt = ts[1] - ts[0]
    ts = np.hstack((ts, ts[-1] + dt))
    ys = np.vstack((ys, ys[-1:, ...]))

    left_index = np.array((t - ts[0]) / dt, dtype=np.int32)
    right_index = left_index + 1
    y_lerp = (ys[left_index] * (ts[right_index] - t) + ys[right_index] * (t - ts[left_index])) / dt

    return y_lerp

lerp_batch = jax.vmap(lerp, in_axes=(None, None, 0), out_axes=0)


def get_aug_rhs_func(state_rhs_func):
    def aug_rhs_func(aug, neg_t, *aug_args):
        y, y_bar, diff_args_bar = aug

        (ys, ts), diff_args = aug_args

        # dt = ts[1] - ts[0]
        # ts = np.hstack((ts, ts[-1] + dt))
        # ys = np.vstack((ys, ys[-1:, ...]))
        # t = -neg_t
        # left_index = np.array((t - ts[0]) / dt, dtype=np.int32)
        # right_index = left_index + 1
        # y_lerp = (ys[left_index] * (ts[right_index] - t) + ys[right_index] * (t - ts[left_index])) / dt

        # Need tests
        y_lerp = lerp(ts, ys, -neg_t)

        y_dot, vjpfun = jax.vjp(lambda y, *diff_args: state_rhs_func(y, -neg_t, *diff_args), y_lerp, *diff_args)
        y_bar_dot, *diff_args_bar_dot = vjpfun(y_bar)

        # If objective function is an integral over time?
        # def obj_func(yf):
        #     xf = yf[:, :3]
        #     center = np.mean(xf, axis=0).reshape(1, dim)
        #     return -np.sum((xf - center)**2)/1e5
        # g_fn = jax.grad(obj_func)
        # y_bar_dot = y_bar_dot + g_fn(y_lerp)

        return (-y_dot, y_bar_dot, *diff_args_bar_dot)

    return aug_rhs_func


def optimize(initials, ts, obj_func, state_rhs_func, bounds=None):
    x_ini, unravel = ravel_pytree(initials)
    aug_rhs_func = get_aug_rhs_func(state_rhs_func)

    obj_vals = []

    def objective(x):
        print(f"\n######################### Evaluating objective value - step {objective.counter}")
        y0, diff_args = unravel(x)
        ys = odeint_ravelled(rk4, state_rhs_func, y0, ts, *diff_args)
        obj_val = obj_func(ys[-1])

        objective.aug0 = (ys[-1], jax.grad(obj_func)(ys[-1]), tree_map(np.zeros_like, diff_args))
        # objective.aug0 = (ys[-1], tree_map(np.zeros_like, ys[-1]), tree_map(np.zeros_like, diff_args))

        objective.diff_args = diff_args
        objective.ys = np.vstack((y0[None, ...], ys))
        objective.counter += 1
        objective.x = [y0, diff_args]
        # print(f"y0 = \n{y0}")
        # print(f"yf = \n{ys[-1]}")
        # print(f"ys = {ys[:, 2, 0]}")
        print(f"obj_val = {obj_val}")
        # print(f"diff_args = {diff_args[0]}")
        obj_vals.append(obj_val)
        return obj_val

    def derivative(x):
        diff_args = objective.diff_args
        aug0 = objective.aug0  
        ys = objective.ys
        
        aug_args = ((ys, ts), diff_args)

        ys_bwd, y_bars, diff_args_bars = odeint_ravelled(rk4, aug_rhs_func, aug0, -ts[::-1], *aug_args)
        # ys_bwd, y_bars, diff_args_bars = odeint_ravelled(rk4, aug_rhs_func, aug0, -ts[-1:-20:-1], *aug_args)

        der_val, _ = ravel_pytree(tree_map(lambda x: x[-1], [y_bars, diff_args_bars]))

        # print(f"ys_bwd[-1] = \n{ys_bwd[-1]}")
        print(f"der_y0 = \n{y_bars[-1]}")
        # print(f"der_diff_args = \n{diff_args_bars[0][-1]}")
        # print(f"der_diff_args = \n{diff_args_bars}")
        # print(f"der = {der_val}")
    
        exit()

        # 'L-BFGS-B' requires the following conversion, otherwise we get an error message saying
        # -- input not fortran contiguous -- expected elsize=8 but got 4
        return onp.array(der_val, order='F', dtype=onp.float64)

    # x = x_ini
    # for i in range(10000):
    #     objective(x)
    #     x = x - 5*1e-4*derivative(x)

    objective.counter = 0
    options = {'maxiter': 1000, 'disp': True}  # CG or L-BFGS-B or Newton-CG or SLSQP
    res = opt.minimize(fun=objective,
                       x0=x_ini,
                       method='SLSQP',
                       jac=derivative,
                       bounds=bounds,
                       callback=None,
                       options=options)

    return objective.x

    # print(obj_vals) 
    # fig = plt.figure()
    # plt.plot(np.arange(len(obj_vals)), obj_vals, linestyle='-', marker='o', color='black')
    # plt.xlabel(r"Optimization step")
    # plt.ylabel(r"$J$")
    # plt.show()


def simulate(initials, ts, state_rhs_func):
    y0, diff_args = initials
    ys = odeint(leapfrog, state_rhs_func, y0, ts, *diff_args)
    return ys


if __name__ == '__main__':
    simulate_odeint()
