import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.experimental import optimizers
import argparse
import os
import matplotlib.pyplot as plt
from .general_utils import shuffle_data, show_contours, profile, d_to_line_segs, sign_to_line_segs
from .data_generator import generate_supervised_data


onp.random.seed(0)
key = jax.random.PRNGKey(0)


parser = argparse.ArgumentParser()                
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--latent_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=1000)  
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--domain_length', type=float, default=2.)
parser.add_argument('--dir', type=str, default='data')
args = parser.parse_args()


def get_ref_seeds(params):
    angles = np.linspace(0, 2 * np.pi, len(params) + 1)[:-1]
    ref_seeds =  np.vstack([params*np.cos(angles), params*np.sin(angles)]).T
    return ref_seeds


def get_ref_seedsAB(params):
    radius_sample_rolled = np.roll(params, -1)
    angles = np.linspace(0, 2 * np.pi, len(params) + 1)[:-1]
    angles_rolled = np.roll(angles, -1)
    seedsA =  np.vstack([params*np.cos(angles), params*np.sin(angles)]).T
    seedsB = np.vstack([radius_sample_rolled*np.cos(angles_rolled), radius_sample_rolled*np.sin(angles_rolled)]).T
    return seedsA, seedsB


def get_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def eval_mass(params):
    seedsA, seedsB = get_ref_seedsAB(params)
    triangle_areas =  1./2 * np.absolute(np.cross(seedsA, seedsB))
    polygon_area = np.sum(triangle_areas)
    triagnle_centroids = 2./3. * 1./2 * (seedsA + seedsB)
    polygon_centroid = np.sum((triangle_areas.reshape(-1, 1) * triagnle_centroids), axis=0) / polygon_area
    triangle_inertias = 1./6. * triangle_areas * (np.sum(seedsA * seedsA, axis=1) +  
        np.sum(seedsA * seedsB, axis=1) + np.sum(seedsB * seedsB, axis=1))
    polygon_inertia = np.sum(triangle_inertias)

    #TODO: Fix the bug of inertia calculation

    return polygon_area, polygon_inertia, polygon_centroid


def reference_to_physical(x1, x2, theta, ref_centroid, ref_points):
    rot = get_rot_mat(theta)
    points_wrt_centroid_initial = ref_points - ref_centroid.reshape(1, -1)
    points_wrt_centroid = points_wrt_centroid_initial @ rot.T
    phy_centroid = np.array([x1, x2])
    phy_points = points_wrt_centroid + phy_centroid
    return phy_points


def get_phy_seeds(params, ref_centroid, x1, x2, theta):
    ref_seeds = get_ref_seeds(params)
    phy_seeds = reference_to_physical(x1, x2, theta, ref_centroid, ref_seeds)
    return phy_seeds

batch_get_phy_seeds = jax.vmap(get_phy_seeds, in_axes=(None, None, 0, 0, 0), out_axes=0)


def eval_sdf(params, ref_centroid, x1, x2, theta, phy_point):
    ref_seedsA, ref_seedsB = get_ref_seedsAB(params)
    phy_seedsA = reference_to_physical(x1, x2, theta, ref_centroid, ref_seedsA)
    phy_seedsB = reference_to_physical(x1, x2, theta, ref_centroid, ref_seedsB)
    ref_pointO = np.array([0., 0.])
    phy_pointO = reference_to_physical(x1, x2, theta, ref_centroid, ref_pointO)
    sign = np.where(np.any(sign_to_line_segs(phy_point, phy_pointO, phy_seedsA, phy_seedsB)), -1., 1.)
    result = np.min(d_to_line_segs(phy_point, phy_seedsA, phy_seedsB)) * sign
    return result

grad_sdf = jax.grad(eval_sdf, argnums=(5))
batch_eval_sdf = jax.vmap(eval_sdf, in_axes=(None, None, None, None, None, 0), out_axes=0)
batch_grad_sdf = jax.vmap(grad_sdf, in_axes=(None, None, None, None, None, 0), out_axes=0)


def single_forward(params, phy_point):
    _, _, polygon_centroid = eval_mass(params)
    return eval_sdf(params, polygon_centroid, polygon_centroid[0], polygon_centroid[1], 0., phy_point)

batch_forward = jax.vmap(single_forward, in_axes=(None, 0), out_axes=0)


@jax.jit
def update(params, opt_state, indices, points, distances):
    value, grads = value_and_grad(loss)(params, indices, points, distances)
    opt_state = opt_update(0, grads, opt_state) 
    return get_params(opt_state), opt_state, value


def loss(params, indices, points, distances):
    batch_points = points[indices]
    batch_distances = distances[indices]
    batch_predicted = batch_forward(params, batch_points).reshape(-1, 1)
    assert batch_predicted.shape == batch_distances.shape
    loss_value = np.sum((batch_predicted - batch_distances)**2)
    params_rolled = np.roll(params, -1)
    reg = np.sum((params - params_rolled)**2)
    return loss_value + 1e-1*reg

opt_init, opt_update, get_params = optimizers.adam(step_size=args.lr)


def train():
    points, distances = generate_supervised_data(args)
    points = np.array(points)
    distances = np.array(distances)

    # boundary_points = np.array(np.load(os.path.join(args.dir, 'numpy/training/boundary_points.npy')))[:100]
    # points = boundary_points[0]
    # distances = np.zeros((len(points), 1))

    args.sample_size = points.shape[0]
    indices = onp.random.permutation(args.sample_size)
    train_indices, test_indices, train_loader, test_loader = shuffle_data(indices, args)

    params = np.ones(args.latent_size)
    # params = np.sqrt(np.sum(points**2, axis=-1))

    opt_state = opt_init(params)

    for epoch in range(args.n_epochs):
        train_loss = 0
        for batch_idx, indices in enumerate(train_loader):
            indices = np.array(indices)
            params, opt_state, loss_value = update(params, opt_state, indices, points, distances)
            train_loss += loss_value

        train_loss_per_sample = train_loss / len(train_loader.dataset)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Per sample loss is {train_loss_per_sample}")

    show_contours(batch_forward, params, 0, args)


def profile_test():
    profile(batch_forward, args)


if __name__ == '__main__':
    # profile_test()
    train()
    plt.show()
