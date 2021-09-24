import numpy as onp
import jax
import jax.numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import time
import glob
import meshio
import os
import matplotlib.pyplot as plt

onp.random.seed(0)
key = jax.random.PRNGKey(0)


def shuffle_data(data, args):
    train_portion = 0.9
    n_samps = len(data)
    n_train = int(train_portion * n_samps)
    inds_train = data[:n_train]
    inds_test = data[n_train:]
    train_data = data[inds_train]
    test_data = data[inds_test]
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    return train_data, test_data, train_loader, test_loader


def show_contours(batch_forward, params, fig_no, args):
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
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])


def profile(forward, args):
    def func(latent_params, batch_points):
        return np.sum(forward(latent_params, batch_points))
    start = time.time()
    # latent_params = generate_radius_samples(num_samps=100, num_division=args.latent_size)[0]
    latent_params = np.load('data/numpy/training/radius_samples.npy')[0]
    batch_points = jax.random.normal(key, (1000, args.dim))
    value, grads = jax.jit(jax.value_and_grad(func))(latent_params, batch_points)
    print(value)
    print(grads)
    end = time.time()
    print(f"Wall time eplapsed: {end - start}")


def clean_folder(directory_path):
    print("Clean folder...")
    files = glob.glob(directory_path)
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to delete {f}, reason: {e}")


def output_vtk_2D(batch_forward, params, step, sample_index, args):
    network_params, latent_params = params
    latent = latent_params[sample_index]

    if sample_index == 1:
        latent = (latent_params[0] + latent_params[1]) / 2

    print(f"Output vtk of sample {sample_index}")
    refinement = 6
    division = onp.power(2, refinement)
    X1D = onp.linspace(-args.domain_length, args.domain_length, division + 1)
    Y1D = onp.linspace(-args.domain_length, args.domain_length, division + 1)
    X_coo, Y_coo = onp.meshgrid(X1D, Y1D, indexing='ij')

    if step == 0:
        clean_folder(os.path.join(args.dir, "vtk/2d/sample_index_{sample_index}/*"))

    cells = []
    for i in range(division):
        for j in range(division):
            cells.append([i*(division + 1) + j, (i + 1)*(division + 1) + j, (i + 1)*(division + 1) + j + 1, i*(division + 1) + j + 1])
    cells = [('quad', onp.array(cells))]
    X_coo_flat = X_coo.flatten()
    Y_coo_flat = Y_coo.flatten()
    Z_coo_flat = onp.zeros_like(X_coo_flat)

    coo = onp.concatenate((X_coo_flat.reshape(-1, 1), Y_coo_flat.reshape(-1, 1)), axis=1)
    repeated_latent = onp.repeat(onp.expand_dims(latent, axis=0), len(coo), axis=0)
    inputs = onp.concatenate((repeated_latent, coo), axis=1)
    solution_flat = onp.array(batch_forward(network_params,  inputs))
    solution_flat = solution_flat.reshape(-1)

    points = onp.stack((X_coo_flat, Y_coo_flat, Z_coo_flat), axis=1)
    point_data = {'u': solution_flat}
    meshio.Mesh(points, cells, point_data=point_data).write(os.path.join(args.dir, f"vtk/2d/sample_index_{sample_index}/u{step}.vtk"))


