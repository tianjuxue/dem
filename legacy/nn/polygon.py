import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.experimental import optimizers
import argparse
import os
import matplotlib.pyplot as plt
from .general_utils import shuffle_data, show_contours, profile
from .data_generator import generate_supervised_data


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