import jax
import jax.numpy as np
from jax_dem.optimizer import optimize
from jax_dem.arguments import args
from jax_dem.dynamics import env_distance_values


def single_bouncing_ball():
    case_name = 'sparse_perfect_ball'
    env_top = args.env_top
    env_bottom = args.env_bottom

    def ini_func():
        ts = np.arange(0., 0.1, 0.2*1e-3)
        # ts = np.arange(0., 0.1, 1e-3)
        radius = 0.5 
        n_objects = 1
        xx = np.array([50, 50, env_bottom + 10*radius]).reshape(-1, 1)
        vv = np.array([0, 0, -2*10*radius/0.1]).reshape(-1, 1)
        q0 = np.ones((1, n_objects))
        y0 = np.concatenate([xx, q0, np.zeros((3, n_objects)), vv, np.zeros((3, n_objects))], axis=0)
        radii = radius * np.ones(y0.shape[1])
        assert np.all(env_distance_values(xx.T) > 0), "Found particle outside of box!"
        return y0, ts, (radii,)

    def obj_func(y_final):
        x = y_final[0:3, :].T
        return x[0, 2]

    optimize(ini_func, obj_func)


if __name__ == '__main__':
    single_bouncing_ball()