import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import argparse
from . import arguments
from .general_utils import d_to_line_segs, sign_to_line_segs

jax.config.update('jax_platform_name', 'cpu')

onp.random.seed(0)
key = jax.random.PRNGKey(0)


def get_angles(num_division):
    return onp.linspace(0, 2 * onp.pi, num_division + 1)[:-1]


def generate_radius_samples(num_samps, num_division=64):
    '''Generate multivariate Gaussian samples.
    Each sample is a vector of radius.

    Returns
    -------
    Numpy array of shape (num_samples, num_division)
    '''

    def kernel(x1, x2):
        '''Periodic kernel
        '''
        sigma = 0.2
        l = 0.4
        p = 2 * onp.pi
        k = sigma**2 * onp.exp(-2 * onp.sin(onp.pi * onp.absolute(x1 - x2) / p)**2 / l**2)
        return k
 
    def mean(x):
        return 1.

    angles = get_angles(num_division)
 
    kernel_matrix = onp.zeros((num_division, num_division))
    mean_vector = onp.zeros(num_division)

    for i in range(num_division):
        mean_vector[i] = mean(angles[i])
        for j in range(num_division):
            kernel_matrix[i][j] = kernel(angles[i], angles[j])

    radius_samples = onp.random.multivariate_normal(mean_vector, kernel_matrix, num_samps)

    assert onp.min(radius_samples) > 0, "Radius must be postive!"
    assert onp.max(radius_samples) < 2, "Radius too large!"

    return radius_samples


def generate_eikonal_points(radius_samples):
    '''For each boundary point in each radius_sample, we generate num_eikonal points at 
    which the Eikonal loss will be evaluated
    
    Returns
    -------
    Numpy array of shape (num_samples, num_division, num_eikonal, dim)
    '''
    num_eikonal = 100
    angles = get_angles(len(radius_samples[0]))
    x = radius_samples * onp.cos(angles)
    y = radius_samples * onp.sin(angles)
    repeated_x = onp.repeat(onp.expand_dims(x, axis=-1), num_eikonal, axis=-1)
    repeated_y =onp.repeat(onp.expand_dims(y, axis=-1), num_eikonal, axis=-1)
    x_samples = onp.random.normal(loc=repeated_x, scale=0.2)
    y_samples = onp.random.normal(loc=repeated_y, scale=0.2)
    eikonal_points = onp.stack([x_samples, y_samples], axis=3)
    return eikonal_points


def compute_boundary_points(radius_samples):
    '''For each boundary point in each radius_sample, we compute the coordinates at 
    which the boundary loss will be evaluated
    
    Returns
    -------
    Numpy array of shape (num_samples, num_division, dim)
    '''
    angles = get_angles(len(radius_samples[0]))
    x = radius_samples * onp.cos(angles)
    y = radius_samples * onp.sin(angles)
    boundary_points = onp.stack([x, y], axis=2)
    return boundary_points


def plot_shape(radius_sample, fig_id=0):
    plt.figure(num=fig_id)
    angles = onp.linspace(0, 2 * onp.pi, len(radius_sample) + 1)[:-1]
    for i in range(len(radius_sample)):
        plt.plot([0., radius_sample[i] * onp.cos(angles[i])], [0., radius_sample[i] * onp.sin(angles[i])], 
            linestyle='-',  linewidth=1, marker='o', markersize=5, color='black')
    plt.axis('equal')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])


def plot_shapes_batch(radius_samples):
    for i in range(len(radius_samples)):
        plot_shape(radius_samples[i], i)


def plot_Eiknoal_points(radius_sample, x_samples_per_shape, y_samples_per_shape):
    plot_shape(radius_sample)
    for i in range(len(x_samples_per_shape)):
        plt.scatter(x_samples_per_shape[i], y_samples_per_shape[i], marker='s', s=2, color='red')


def circleSDF(x, y):
    return onp.sqrt(x**2 + y**2) - 1


def squareSDF(x, y):
    points = np.vstack([x, y]).T
    square_corners = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
    square_corners_rolled = np.roll(square_corners, 1, axis=0)
    sdf = onp.zeros_like(x)
    for i in range(len(sdf)):
        sign = np.where(np.any(sign_to_line_segs(points[i], square_corners, square_corners_rolled)), -1., 1.)
        sdf[i] = np.min(d_to_line_segs(points[i], square_corners, square_corners_rolled)) * sign
    return sdf


def generate_circle_boundary_data():
    n = 1000
    theta = onp.random.uniform(0., 2*onp.pi, n)
    coo = onp.stack([onp.cos(theta), onp.sin(theta)]).T
    sdf = onp.zeros((n, 1))
    return coo, sdf


def generate_supervised_data(args):
    n = 1000
    xy_min = [-args.domain_length, -args.domain_length]
    xy_max = [args.domain_length, args.domain_length]
    coo = onp.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
    sdf = squareSDF(coo[:, 0], coo[:, 1]).reshape(-1, 1)
    # samples = onp.concatenate((coordinates, targets.reshape(-1, 1)), axis=1)
    return coo, sdf


def main(args):
    radius_samples = generate_radius_samples(num_samps=1000)
    boundary_points = compute_boundary_points(radius_samples)
    eikonal_points = generate_eikonal_points(radius_samples)

    onp.save('data/numpy/training/radius_samples.npy', radius_samples)
    onp.save('data/numpy/training/boundary_points.npy', boundary_points)
    onp.save('data/numpy/training/eikonal_points.npy', eikonal_points)
    
    # Visualization
    sample_index = 0
    plot_Eiknoal_points(radius_samples[sample_index], eikonal_points[sample_index, ..., 0], eikonal_points[sample_index, ..., 1])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_length', type=float, default=2.)
    args = parser.parse_args()
    main(args)
