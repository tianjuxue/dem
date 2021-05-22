import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Relu, Sigmoid, Softplus, Selu, Tanh, Identity, FanOut, FanInConcat
from jax.nn.initializers import normal
import matplotlib.pyplot as plt
import glob
import meshio
import os
import sys
import argparse
from functools import partial
import datetime
import json
from .data_generator import plot_Eiknoal_points
from .general_utils import shuffle_data, profile, clean_folder, output_vtk_2d

# jax.config.update('jax_platform_name', 'cpu')

onp.random.seed(0)
key = jax.random.PRNGKey(0)

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=3)

parser = argparse.ArgumentParser()

# NN parameters
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--radius_size', type=int, default=64)
parser.add_argument('--latent_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--width_hidden', type=int, default=128)
parser.add_argument('--skip', action='store_true', default=True)
parser.add_argument('--n_skip', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=2)
parser.add_argument('--activation', choices=['tanh', 'selu', 'relu'], default='tanh')
parser.add_argument('--n_epochs', type=int, default=10)  
parser.add_argument('--weight_latent', type=float, default=0.001)
parser.add_argument('--weight_eikonal', type=float, default=0.01)
parser.add_argument('--weight_boundary', type=float, default=1.)
parser.add_argument('--weight_supervised', type=float, default=1.)
parser.add_argument('--lr', type=float, default=2*1e-3)

# misc
parser.add_argument('--domain_length', type=float, default=2.)
parser.add_argument('--dir', type=str, default='data')

args = parser.parse_args()


def get_mlp():
    if args.activation == 'selu':
        act_fun = Selu
    elif args.activation == 'tanh':
        act_fun = Tanh
    elif args.activation == 'relu':
        act_fun = Relu
    else:
        raise ValueError(f"Invalid activation function {args.activation}.")

    layers_hidden = []
    for _ in range(args.n_hidden):
        # layers_hidden.extend([Dense(args.width_hidden), act_fun])
        layers_hidden.extend([Dense(args.width_hidden)])
    layers_hidden.append(act_fun)
    layers_hidden.append(Dense(1))

    if args.skip:
        layers_skip = []
        for _ in range(args.n_skip):
            # layers_skip.extend([Dense(args.width_hidden), act_fun])
            layers_skip.extend([Dense(args.width_hidden)])
        layers_skip.append(act_fun)
        layers_skip.append(Dense(args.width_hidden - args.latent_size - args.dim))
        mlp = stax.serial(FanOut(2),  stax.parallel(Identity, stax.serial(*layers_skip)), FanInConcat(), stax.serial(*layers_hidden))
    else:
        mlp = stax.serial(*layers_hidden)

    return mlp


def compute_supervised_points(boundary_points, args):
    '''Compute supervised signed distance
    
    Returns
    -------
    Numpy array of shape (num_samples, num_supvervision, dim)
    Numpy array of shape (num_samples, num_supvervision, 1)
    '''
    p0 = onp.array([0., 0.])
    p1 = onp.array([-args.domain_length, -args.domain_length])
    p2 = onp.array([-args.domain_length, args.domain_length])
    p3 = onp.array([args.domain_length, -args.domain_length])
    p4 = onp.array([args.domain_length, args.domain_length])
    points = onp.array([p0, p1, p2, p3, p4])
    # points = onp.array([p0])

    supervised_points = onp.repeat(onp.expand_dims(points, axis=0), len(boundary_points), axis=0)

    # compute signed distance
    signed_distance = onp.stack([onp.min(onp.sqrt(onp.sum((p - boundary_points)**2, axis=-1)), axis=-1) for p in points], axis=-1)
    signed_distance[:, 0] *= -1
    signed_distance = signed_distance.reshape(signed_distance.shape[0], signed_distance.shape[1], 1)

    return np.array(supervised_points), np.array(signed_distance)


@partial(jax.jit, static_argnums=(7,))
def update(params, opt_state, indices, boundary_points, eikonal_points, supervised_points, supervised_distance, dim):
    value, grads = value_and_grad(loss)(params, indices, boundary_points, eikonal_points, supervised_points, supervised_distance, dim)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value


def batch_forward_wrapper(batch_forward):
    def single_forward(network_params, latent, points):
        input_single = np.concatenate((latent, points)).reshape(1, -1)
        return batch_forward(network_params, input_single)[0][0]
    return single_forward


def eikonal_loss(network_params, latent, points):
    grad_value = grad(batch_forward_wrapper(batch_forward), argnums=2)(network_params, latent, points)
    penalty = (np.linalg.norm(grad_value) - 1.)**2
    return penalty


def append_latent(batch_latent_params, batch_points):
    '''
    Parameters
    ----------
    batch_latent_params: numpy array of shape (batch, latent_size)
    batch_points: numpy array of shape (batch, N, dim)

    Returns
    -------
    Numpy array of shape (N * batch, latent_size + dim)
    '''
    assert len(batch_latent_params) == len(batch_points), "Batch size should be equal!"
    inputs_points = batch_points.reshape(batch_points.shape[0], -1, batch_points.shape[-1])
    inputs_latent = np.repeat(np.expand_dims(batch_latent_params, axis=1), inputs_points.shape[1], axis=1)
    inputs = np.concatenate((inputs_latent, inputs_points), axis=-1)
    inputs = inputs.reshape(-1, inputs.shape[-1])
    return inputs


def get_eikonal_samples(batch_latent_params, batch_eikonal_points, uniform_flag, dim):
    n_satellite = 10
    if uniform_flag:
        xy_min = [-2, -2]
        xy_max = [2, 2]
        batch_uniform_inputs = onp.random.uniform(low=xy_min, 
            high=xy_max, size=(batch_eikonal_points.shape[0], batch_eikonal_points.shape[1] * n_satellite , dim))
        uniform_inputs = append_latent(batch_latent_params, batch_uniform_inputs)
        return uniform_inputs
    else:
        eikonal_inputs = append_latent(batch_latent_params, batch_eikonal_points[...,:n_satellite,:])
        return eikonal_inputs

    # eikonal_inputs = np.concatenate((eikonal_inputs, uniform_inputs), axis=0)


def loss(params, indices, boundary_points, eikonal_points, supervised_points, supervised_distance, dim):
    network_params, latent_params = params

    batch_latent_params = latent_params[indices]
    batch_eikonal_points = eikonal_points[indices]
    batch_boundary_points = boundary_points[indices]
    batch_supervised_points = supervised_points[indices]
    batch_supervised_distance = supervised_distance[indices]

    eikonal_inputs = get_eikonal_samples(batch_latent_params, batch_eikonal_points, False, dim)
    boundary_inputs = append_latent(batch_latent_params, batch_boundary_points)
    supervised_inputs = append_latent(batch_latent_params, batch_supervised_points)
    supervised_outputs = batch_supervised_distance.reshape(-1, supervised_distance.shape[-1])

    latent_loss_value = np.sum(batch_latent_params**2)
    eikonal_loss_batch = vmap(eikonal_loss, in_axes=(None, 0, 0), out_axes=0)
    eikonal_loss_value =  np.sum(eikonal_loss_batch(network_params, eikonal_inputs[:, :-dim], eikonal_inputs[:, -dim:]))
    boundary_loss_value = np.sum(batch_forward(network_params, boundary_inputs)**2)
    supervised_loss_value = np.sum((batch_forward(network_params, supervised_inputs) - supervised_outputs)**2)

    return 0.001 * latent_loss_value + 0.01 * eikonal_loss_value + boundary_loss_value + 1 * supervised_loss_value


def forward(latent_params, batch_points, network_params):
    inputs = append_latent(np.expand_dims(latent_params, axis=0), np.expand_dims(batch_points, axis=0))
    return batch_forward(network_params, inputs)


opt_init, opt_update, get_params = optimizers.adam(step_size=args.lr)
init_random_params, batch_forward = get_mlp()


def train():
    # Load training data and shuffle indices
    boundary_points = np.array(np.load(os.path.join(args.dir, 'numpy/training/boundary_points.npy')))[:100]
    eikonal_points = np.array(np.load(os.path.join(args.dir, 'numpy/training/eikonal_points.npy')))[:100]
    supervised_points, supervised_distance = compute_supervised_points(boundary_points, args)
    args.sample_size = boundary_points.shape[0]
    indices = onp.random.permutation(args.sample_size)
    train_indices, test_indices, train_loader, test_loader = shuffle_data(indices, args)

    now = datetime.datetime.now().strftime('%s')
    os.mkdir(os.path.join(args.dir, f'numpy/model/{now}'))

    output_shape, network_params = init_random_params(key, (-1, args.latent_size + args.dim))
    latent_params = np.zeros((args.sample_size, args.latent_size))
    params = [network_params, latent_params]
    opt_state = opt_init(params)
    train_loss_record = {}
    for epoch in range(args.n_epochs):
        train_loss = 0
        for batch_idx, indices in enumerate(train_loader):
            indices = np.array(indices)
            params, opt_state, loss_value = update(params, opt_state, indices, boundary_points, eikonal_points, supervised_points, supervised_distance, args.dim)
            train_loss += loss_value

        train_loss_per_sample = train_loss / len(train_loader.dataset)
        train_loss_record[f'epoch_{epoch}'] = f'{float(train_loss_per_sample):.5f}'

        if epoch % 3 == 0:
            print(f"Epoch {epoch}: Per sample loss is {train_loss_per_sample}")
            output_vtk_2d(batch_forward, params, epoch, 0, args)
            output_vtk_2d(batch_forward, params, epoch, 1, args)
            np.save(os.path.join(args.dir, f'numpy/model/{now}/params_at_epoch_{epoch}.npy'), params)

    with open(os.path.join(args.dir, f'numpy/model/{now}/args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    with open(os.path.join(args.dir, f'numpy/model/{now}/train_loss.txt'), 'w') as f:
        json.dump(train_loss_record, f, indent=2)


def profile_test():
    output_shape, network_params = init_random_params(key, (-1, args.latent_size + args.dim))
    profile(partial(forward, network_params=network_params), args)


if __name__ == '__main__':
    train() 
    # profile_test()
