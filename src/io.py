import numpy as onp
import jax
import jax.numpy as np
import time
import glob
import meshio
import os
import matplotlib.pyplot as plt


def output_vtk_3D_shape(points, connectivity, file_path):
    cells = [("triangle", connectivity)]
    mesh = meshio.Mesh(points, cells)
    mesh.write(file_path)


def output_vtk_3D_field(args, scalar_func, file_path):
    refinement = 6
    division = onp.power(2, refinement)
    tmp = 3 * [onp.linspace(-args.domain_length, args.domain_length, division + 1)]
    X_coo, Y_coo, Z_coo = onp.meshgrid(*tmp, indexing='ij')

    cells = []
    for i in range(division):
        for j in range(division):
            for k in range(division):
                N = division + 1
                cells.append([i * N**2 + j * N + k, 
                             (i + 1) * N**2 + j * N + k, 
                             (i + 1) * N**2 + (j + 1) * N + k, 
                             i * N**2 + (j + 1) * N + k,
                             i * N**2 + j * N + k + 1, 
                             (i + 1) * N**2 + j * N + k + 1, 
                             (i + 1) * N**2 + (j + 1) * N + k + 1, 
                             i * N**2 + (j + 1) * N + k + 1])
    cells = [('hexahedron', onp.array(cells))]
    X_coo_flat = X_coo.flatten()
    Y_coo_flat = Y_coo.flatten()
    Z_coo_flat = Z_coo.flatten()

    points = onp.stack((X_coo_flat, Y_coo_flat, Z_coo_flat), axis=1)
    solutions = scalar_func(points)
    point_data = {'u': solutions}
    meshio.Mesh(points, cells, point_data=point_data).write(file_path)


def plot_energy(energy, file_path):
    plt.figure(num=10, figsize=(6, 6))
    plt.plot(20*np.arange(1, len(energy) + 1, 1), energy, marker='o',  markersize=2, linestyle="-", linewidth=1, color='blue')
    plt.xlabel("Time steps")
    plt.ylabel("Energy")
    plt.savefig(file_path)