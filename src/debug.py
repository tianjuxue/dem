import numpy as onp
import jax
import jax.numpy as np
import time
from jax import grad, jit, vmap, value_and_grad
from functools import partial
from .shape3d import generate_template_object, batch_eval_sdf_helper, batch_grad_sdf_helper

dim = 3

def main():
    object_3D = generate_template_object('sphere', 6)
    vertices_oriented = object_3D.get_oriented_vertices()
    origin = np.array([0., 0., 0.])

    for i in range(5):

        key = jax.random.PRNGKey(0)
        test_points = jax.random.uniform(key, shape=(125*2*122, dim), minval=-3., maxval=3.)
        break1 = time.time()
        evals = batch_eval_sdf_helper(vertices_oriented, origin, test_points).block_until_ready()
        break2 = time.time()

        print(evals.shape)
        print(f"time: {break2 - break1}")
 

if __name__ == '__main__':
    main()
