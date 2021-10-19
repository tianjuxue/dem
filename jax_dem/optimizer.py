import jax
import jax.numpy as np
import numpy as onp
from jax import linear_util as lu
from jax.flatten_util import ravel_pytree
from functools import partial
from jax.tree_util import tree_map
from jax_dem.io import vedo_plot
import scipy.optimize as opt
import matplotlib.pyplot as plt


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
        k4 = h * f(y_prev + k3, t_prev + h, *diff_args)
        y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t_crt), y

    # def step(state, t_crt):
    #     y_prev, t_prev = state
    #     h = t_crt - t_prev
    #     y = y_prev + h * f(y_prev, t_prev, *diff_args)
    #     return (y, t_crt), y


    # _, ys = jax.lax.scan(step, (y0, ts[0]), ts[1:])

    ys = []
    energy = []
    state = (y0, ts[0])
    for (i, t_crt) in enumerate(ts[1:]):
        state, y = step(state, t_crt)
        # ys.append(y)
        # print("break")
        if i % 20 == 0:
            # e = compute_energy(radii, y)
            # print(f"\nstep {i}, total energy={e}, quaternion square sum: {np.sum(y[3:7]**2)}")
            print(f"step {i}")
            if not np.all(np.isfinite(y)):
                print(f"Found np.inf or np.nan in y - stop the program")  
                # print(y)              
                exit()
        ys.append(y)
            # energy.append(e)
    # energy = np.array(energy)
    ys = np.array(ys)

    return ys


def get_aug_rhs_func(state_rhs_func):
    def aug_rhs_func(aug, neg_t, *aug_args):
        y, y_bar, diff_args_bar = aug

        (ys, ts), diff_args = aug_args

        # Assumption: Uniform time step
        dt = ts[1] - ts[0]
        ts = np.hstack((ts, ts[-1] + dt))
        ys = np.vstack((ys, ys[-1:, ...]))

        assert len(ts) == len(ys), f"ts.shape = {ts.shape}, ys.shape = {ys.shape}"

        t = -neg_t
        left_index = np.array((t - ts[0]) / dt, dtype=np.int32)
        right_index = left_index + 1
        y_lerp = (ys[left_index] * (ts[right_index] - t) + ys[right_index] * (t - ts[left_index])) / dt

        # print(f"left_index = {left_index}, right_index = {right_index}, t = {t}")
        # print(f"y_lerp=\n{y_lerp}")

        y_dot, vjpfun = jax.vjp(lambda y, *diff_args: state_rhs_func(y, -neg_t, *diff_args), y_lerp, *diff_args)
        return (-y_dot, *vjpfun(y_bar))

    return jax.jit(aug_rhs_func)


def optimize(initials, ts, obj_func, state_rhs_func, bounds=None):
    x_ini, unravel = ravel_pytree(initials)
    aug_rhs_func = get_aug_rhs_func(state_rhs_func)

    obj_vals = []

    def objective(x):
        print(f"\n######################### Evaluating objective value - step {objective.counter}")
        y0, diff_args = unravel(x)
        ys = odeint(state_rhs_func, y0, ts, *diff_args)
        obj_val = obj_func(ys[-1])
        objective.aug0 = (ys[-1], jax.grad(obj_func)(ys[-1]), tree_map(np.zeros_like, diff_args))
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

        ys_bwd, ys_bar, diff_args_bar = odeint(aug_rhs_func, aug0, -ts[::-1], *aug_args)
        der_val, _ = ravel_pytree(tree_map(lambda x: x[-1], [ys_bar, diff_args_bar]))

        # fig = plt.figure()
        # plt.plot(ts[-2::-1], ys_bar[:, 0, 0], linestyle='-', marker='o', color='red')
        # plt.show()

        # print(f"ys_bwd[-1] = \n{ys_bwd[-1]}")
        # print(f"der_y0 = \n{ys_bar[-1]}")
        # print(f"der_diff_args = \n{diff_args_bar[0][-1]}")
        print(f"der_diff_args = \n{diff_args_bar}")
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
    ys = odeint(state_rhs_func, y0, ts, *diff_args)

    # plot_energy(energy, f'data/pdf/energy_{object_name}.pdf')

    return ys


if __name__ == '__main__':
    simulate_odeint()
