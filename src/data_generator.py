import numpy as onp
import matplotlib.pyplot as plt
import argparse
from . import arguments
onp.random.seed(0)


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


def distance_function_line_segement(P, A=[-1, 0], B=[1, 0]):     
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]
  
    BP = [None, None]
    BP[0] = P[0] - B[0]
    BP[1] = P[1] - B[1]

    AP = [None, None]
    AP[0] = P[0] - A[0]
    AP[1] = P[1] - A[1]
  
    AB_BP = AB[0] * BP[0] + AB[1] * BP[1]
    AB_AP = AB[0] * AP[0] + AB[1] * AP[1]
  
    y = P[1] - B[1]
    x = P[0] - B[0]
    df1 = onp.sqrt(x**2 + y**2) 

    y = P[1] - A[1]
    x = P[0] - A[0]
    df2 = onp.sqrt(x**2 + y**2)

    x1 = AB[0]
    y1 = AB[1]
    x2 = AP[0]
    y2 = AP[1]
    mod = onp.sqrt(x1**2 + y1**2)
    df3 = onp.absolute(x1 * y2 - y1 * x2) / mod

    if AB_BP > 0:
        df = df1
    elif AB_AP < 0:
        df = df2
    else:
        df = df3

    return df


def squareSDF(x, y):
    A = [-1, -1]
    B = [-1, 1]
    C = [1, 1]
    D = [1, -1]
    sdf = onp.zeros_like(x)
    for i in range(len(sdf)):
        d1 = distance_function_line_segement([x[i], y[i]], A, B)
        d2 = distance_function_line_segement([x[i], y[i]], B, C)
        d3 = distance_function_line_segement([x[i], y[i]], C, D)
        d4 = distance_function_line_segement([x[i], y[i]], D, A)  
        d = onp.min(onp.array([d1, d2, d3, d4]))  
        if x[i] < 1 and x[i] > -1 and y[i] < 1 and y[i] > -1:
            d = -d
        sdf[i] = d

    return sdf


def generate_boundary_data():
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
    sdf = circleSDF(coo[:, 0], coo[:, 1]).reshape(-1, 1)
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