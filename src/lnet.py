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
from .general_utils import shuffle_data, show_contours
from .data_generator import generate_supervised_data


jax.config.update('jax_platform_name', 'cpu')

onp.random.seed(0)
key = jax.random.PRNGKey(0)

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=3)


parser = argparse.ArgumentParser()
parser.add_argument('--n_layers', type=int, default=20)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=1000)  
parser.add_argument('--lr', type=float, default=5*1e-3)
parser.add_argument('--domain_length', type=float, default=2.)
parser.add_argument('--dir', type=str, default='data')
args = parser.parse_args()


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


def get_mlp():
    layers = [Dense(args.dim, W_init=orthonormal()), Sort()] * args.n_layers
    # layers = [Dense(128), Tanh] * 3
    layers.append(Dense(1, W_init=orthonormal()))
    mlp = stax.serial(*layers)
    return mlp

@jit
def update(params, opt_state, indices, points, distances, w1, w2):
    value, grads = value_and_grad(loss)(params, indices, points, distances, w1, w2)
    opt_state = opt_update(0, grads, opt_state) 
    return get_params(opt_state), opt_state, value


def loss(params, indices, points, distances, w1, w2):
    batch_points = points[indices]
    batch_distances = distances[indices]
    batch_predicted = batch_forward(params, batch_points)
    # Question: Why Jax gives (array1 - array2) the shape of (64, 64)?
    # array1 has shape (64, 1) and is of <class 'jax.interpreters.ad.JVPTracer'>
    # array2 has shape (64,) and is of <class 'jaxlib.xla_extension.DeviceArray'>
     assert batch_predicted.shape == batch_distances.shape
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


opt_init, opt_update, get_params = optimizers.adam(step_size=args.lr)
init_random_params, batch_forward = get_mlp()


def train():
    points, distances = generate_supervised_data(args)
    points = np.array(points)
    distances = np.array(distances)

    args.sample_size = points.shape[0]
    indices = onp.random.permutation(args.sample_size)
    train_indices, test_indices, train_loader, test_loader = shuffle_data(indices, args)

    output_shape, params = init_random_params(key, (-1, args.dim))
    opt_state = opt_init(params)
    for epoch in range(args.n_epochs):
        train_loss = 0
        for batch_idx, indices in enumerate(train_loader):
            indices = np.array(indices)
            params, opt_state, loss_value = update(params, opt_state, indices, points, distances, w1=1., w2=0.5)
            train_loss += loss_value

        train_loss_per_sample = train_loss / len(train_loader.dataset)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Per sample loss is {train_loss_per_sample}")
            print(f"Orthonormal regualization value is {orthnorm_reg(params)}")

    show_contours(batch_forward, params, 0, args)


if __name__ == '__main__':
    train()
    plt.show()