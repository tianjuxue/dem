import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Tanh, Sigmoid, Identity
import matplotlib.pyplot as plt
import argparse
import glob
import meshio
import os
import sys
import argparse
from functools import partial
import datetime
import json
import scipy.linalg as spla
from .general_utils import shuffle_data
from .data_generator import generate_supervised_data, generate_boundary_data
from .deepSDF import compute_supervised_points


jax.config.update('jax_platform_name', 'cpu')

onp.random.seed(0)
key = jax.random.PRNGKey(0)

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=3)


def orthonormal(stddev=1e-2, dtype=np.float32):
    def init(key, shape, dtype=dtype):
        A = jax.random.normal(key, shape, dtype) * stddev
        Q, R = spla.qr(A)
        return Q[:, :shape[1]]
    return init


def Sort():
    """Does Jax have a sort activation function?"""
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: np.sort(inputs)
    return init_fun, apply_fun


def get_mlp(args):
    layers = [Dense(args.dim, W_init=orthonormal()), Sort()] * args.n_layers
    # layers = [Dense(128), Tanh] * 3
    layers.append(Dense(1, W_init=orthonormal()))
    mlp = stax.serial(*layers)
    return mlp


def train(args):

    @jit
    def update(params, opt_state, indices, points, distances, w1, w2):
        value, grads = value_and_grad(loss)(params, indices, points, distances, w1, w2)
        opt_state = opt_update(0, grads, opt_state) 
        return get_params(opt_state), opt_state, value


    def loss(params, indices, points, distances, w1, w2):
        bacth_points = points[indices]
        batch_distances = distances[indices]
        batch_predicted = batch_forward(params, bacth_points)
        # Question: Why Jax gives (array1 - array2) the shape of (64, 64)?
        # array1 has shape (64, 1) and is of <class 'jax.interpreters.ad.JVPTracer'>
        # array2 has shape (64,) and is of <class 'jaxlib.xla_extension.DeviceArray'>
        loss_value = np.sum((batch_predicted - batch_distances)**2)
        orth_reg_value = orthnorm_reg(params)
        return w1 * loss_value + w2 * orth_reg_value


    def orthnorm_reg(params):
        beta = 0.5
        reg = 0.
        for param in params:
            if len(param) != 0:
                W, b = param
                reg += beta/2 * np.sum((W.T @ W - np.eye(W.shape[1]))**2)
        return reg


    def plot(params, fig_no):
        xgrid = np.linspace(-args.domain_length, args.domain_length, 100)
        x1, x2 = np.meshgrid(xgrid, xgrid)
        xx = np.vstack([x1.ravel(), x2.ravel()]).T
        out = np.reshape(batch_forward(params, xx), x1.shape)
        plt.figure(num=fig_no, figsize=(8, 8))
        plt.contourf(x1, x2, out, levels=50, cmap='seismic')
        plt.colorbar()
        contours = np.linspace(-1., 1., 11)
        plt.contour(x1, x2, out, contours, colors=['black']*len(contours))
        plt.contour(x1, x2, out, [0.], colors=['red'])
        plt.axis('equal')
        x_origin = np.array([[0., 0.]])
        print(batch_forward(params, x_origin))


    points, distances = generate_supervised_data(args)
    points = np.array(points)
    distances = np.array(distances)

    args.sample_size = points.shape[0]
    indices = onp.random.permutation(args.sample_size)
    train_indices, test_indices, train_loader, test_loader = shuffle_data(indices, args)

    opt_init, opt_update, get_params = optimizers.adam(step_size=args.lr)
  
    init_random_params, batch_forward = get_mlp(args)
    output_shape, params = init_random_params(key, (-1, args.dim))
    opt_state = opt_init(params)


    for epoch in range(args.n_epochs):
        train_loss = 0
        for batch_idx, indices in enumerate(train_loader):
            indices = np.array(indices)
            params, opt_state, loss_value = update(params, opt_state, indices, points, distances, w1=1., w2=2.)
            train_loss += loss_value

        train_loss_per_sample = train_loss / len(train_loader.dataset)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Per sample loss is {train_loss_per_sample}")
            print(f"Orthonormal regualization value is {orthnorm_reg(params)}")

    plot(params, 0)


    boundary_points = np.array(np.load(os.path.join(args.dir, 'numpy/training/boundary_points.npy')))[:100]
    points1 = boundary_points[0]
    distances1 = np.zeros((len(points1), 1))
    supervised_points, supervised_distance = compute_supervised_points(boundary_points, args)
    points2 = supervised_points[0]
    distances2 = supervised_distance[0]
    points = np.concatenate([points1, points2], axis=0)
    distances = np.concatenate([distances1, distances2], axis=0)

    points = points1
    distances = distances1

    # points, distances = generate_boundary_data()
    # points = np.array(points)
    # distances = np.array(distances)

    args.sample_size = points.shape[0]
    indices = onp.random.permutation(args.sample_size)
    train_indices, test_indices, train_loader, test_loader = shuffle_data(indices, args)

    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
    opt_state = opt_init(params)

    for epoch in range(3 * args.n_epochs):
        train_loss = 0
        for batch_idx, indices in enumerate(train_loader):
            indices = np.array(indices)
            params, opt_state, loss_value = update(params, opt_state, indices, points, distances, w1=1, w2=0)
            train_loss += loss_value

        eps = 1.
        tol = 1e-3
        step = 0
        while eps > tol:
            params, opt_state, loss_value = update(params, opt_state, indices, points, distances, w1=0, w2=100)
            eps = orthnorm_reg(params)
            step += 1
            # print(f"eps={eps}")

        train_loss_per_sample = train_loss / len(train_loader.dataset)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Per sample loss is {train_loss_per_sample}")
            print(f"Orthonormal regualization value is {eps}")
            print(f"Total step of regualization {step}")



    plot(params, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=20)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=1000)  
    parser.add_argument('--lr', type=float, default=5*1e-3)
    parser.add_argument('--domain_length', type=float, default=2.)
    parser.add_argument('--dir', type=str, default='data')
    args = parser.parse_args()
    train(args)
    plt.show()