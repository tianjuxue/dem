'''
This is a failed attempt to implement the following paper
Chater, Mario, Angxiu Ni, and Qiqi Wang. 
"Simplified least squares shadowing sensitivity analysis for chaotic ODEs and PDEs." 
Journal of Computational Physics 329 (2017): 126-140.

I realized that scipy function solve_bvp is not the right way to solve the
boundary value problem we have.

Also see 
Wang, Qiqi, Rui Hu, and Patrick Blonigan. 
"Least squares shadowing sensitivity analysis of chaotic limit cycle oscillations." 
Journal of Computational Physics 267 (2014): 210-224.
In the paper, they unrolled in time and dealt with a giant matrix like Equation (18).

Tianju
10/22/2021
'''
import jax
import jax.numpy as np
from functools import partial
from jax.experimental.ode import odeint
import jax.numpy as np
import matplotlib.pyplot as plt
from jax_dem.optimizer import odeint, rk4, lerp_batch
from scipy.integrate import solve_bvp


def Lorenz_state_rhs_func(state, t, rho, sigma, beta):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def window(r):
    return np.sin(np.pi*r)**4


def obj_func(y):
    return y[2]

obj_func_batch = jax.vmap(obj_func)


def get_bvp_func(state_rhs_func, ys_fixed, ts_fixed, *args):

    assert len(ys_fixed.shape) == 2
    assert len(ys_fixed) == len(ts_fixed)

    T = ts_fixed[-1] - ts_fixed[0]
    n_vars = ys_fixed.shape[1]

    def w_rhs_func(w, y, t):
        _, rhs = jax.jvp(lambda y: state_rhs_func(y, t, *args), (y,), (w,))
        return rhs

    def v_rhs_func(w, v, y, t):
        _, vjpfun = jax.vjp(lambda y: state_rhs_func(y, t, *args), y)
        return -vjpfun(v)[0] -w -jax.grad(obj_func)(y)*window(t/T)

    w_rhs_func_vmap = jax.vmap(w_rhs_func, in_axes=(0, 0, 0), out_axes=0)
    v_rhs_func_vmap = jax.vmap(v_rhs_func, in_axes=(0, 0, 0, 0), out_axes=0)

    @jax.jit
    def func(ts, wvs):
        ws = wvs[:n_vars, :].T
        vs = wvs[n_vars:, :].T
        ys = lerp_batch(ts_fixed, ys_fixed, ts)
        w_rhs = w_rhs_func_vmap(ws, ys, ts)
        v_rhs = v_rhs_func_vmap(ws, vs, ys, ts)
        wv_rhs = np.hstack((w_rhs, v_rhs))
        return wv_rhs.T

    def bc(wv_a, wv_b):
        return np.hstack((wv_a[n_vars:], wv_b[n_vars:]))

    return func, bc


def bvp():
    rho = 28.
    sigma = 10.
    beta = 8./3
    y0 = np.array([1., 1., 1.])
    dt = 0.02
    N1 = 500
    N2 = N1 + 2501
    ts_full = np.arange(0., dt*N2, dt)
    ys_full = odeint(rk4, Lorenz_state_rhs_func, y0, ts_full, rho, sigma, beta)
    ts_fixed = ts_full[N1:]
    ys_fixed = ys_full[N1 - 1:]
    ts_fixed = ts_fixed - ts_fixed[0]

    func, bc = get_bvp_func(Lorenz_state_rhs_func, ys_fixed, ts_fixed, rho, sigma, beta)
    ts_ini = np.linspace(ts_fixed[0], ts_fixed[-1], 1001)
    wvs_ini = np.zeros((2*ys_fixed.shape[1], len(ts_ini)))

    res = solve_bvp(func, bc, ts_ini, wvs_ini, max_nodes=len(ts_fixed), verbose=2)

    # plot_3d_path(res.y[3:, :].T)

    averaged_value(ys_fixed, ts_fixed)
    averaged_grad(ts_fixed, ys_fixed, Lorenz_state_rhs_func, res.y[3:, :].T, res.x, rho, sigma, beta)


def averaged_value(ys, ts):
    tmp = obj_func_batch(ys)
    print(np.sum(tmp)*(ts[1] - ts[0])/(ts[-1] - ts[0]))


def averaged_grad(ts_fixed, ys_fixed, state_rhs_func, vs, ts, *args):
    ys = lerp_batch(ts_fixed, ys_fixed, ts)

    def helper(v, y, t):
        _, vjpfun = jax.vjp(lambda *args: state_rhs_func(y, t, *args), *args)
        return vjpfun(v)[0]

    helper_vmap = jax.vmap(helper, in_axes=(0, 0, 0), out_axes=0)
    tmp = helper_vmap(vs, ys, ts)
    print(np.sum(tmp)*(ts[1] - ts[0])/(ts[-1] - ts[0]))
 

def plot_3d_path(ys):
    fig = plt.figure(figsize=(6, 4), dpi=150)
    ax = fig.gca(projection='3d')
    x0, x1, x2 = ys.T
    ax.plot(x0, x1, x2, lw=0.5, color='b')
    plt.show()

 
if __name__ == '__main__':
    bvp()
