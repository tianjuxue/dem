import jax
import jax.numpy as np
import numpy as onp
from jax_dem.dynamics import compute_sphere_inertia_tensors
from jax_dem.arguments import args
from jax_dem.utils import rotate_vector, rotate_tensor 
dim = args.dim


def initialize_states():
    radius = 0.5
    spacing = np.linspace(0.5, 2.5, 3)
    n_objects = len(spacing)**3
    x1, x2, x3 = np.meshgrid(*([spacing]*3), indexing='ij')
    x0 = np.vstack((x1.reshape(-1), x2.reshape(-1), x3.reshape(-1))).T
    key = jax.random.PRNGKey(seed=0)
    perturb = jax.random.uniform(key, (n_objects, dim), np.float32, -0.5*radius, 0.5*radius)
    x0 += perturb
    radii = radius * np.ones(n_objects)
    return x0, radii


def run():
    key = jax.random.PRNGKey(seed=0)
    x0, radii = initialize_states()
    object_inertia, _, _ = compute_object_inertia_tensor(x0, radii)
    print(f"before rotation: \n{object_inertia}")
    unit_vec = jax.random.normal(key, (dim,))
    unit_vec /= np.linalg.norm(unit_vec)
    q = np.hstack((np.array([np.cos(np.pi/8)]), unit_vec * np.sin(np.pi/8)))
    x_rotated = rotate_vector(q, x0)
    object_inertia_rotated1, _, _ = compute_object_inertia_tensor(x_rotated, radii)
    object_inertia_rotated2 = rotate_tensor(q, object_inertia)
    print(f"after rotation: \n{object_inertia_rotated1}")
    print(f"after rotation: \n{object_inertia_rotated2}")


if __name__ == '__main__':
    run()
