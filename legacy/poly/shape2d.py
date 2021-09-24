import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax.experimental import optimizers
import argparse
import os
import matplotlib.pyplot as plt


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


def d_to_line_seg(P, A, B):
    '''Distance of a point P to a line segment AB'''
    AB = B - A
    BP = P - B
    AP = P - A
    AB_BP = np.dot(AB, BP)
    AB_AP = np.dot(AB, AP)
    mod = np.sqrt(np.sum(AB**2))
    tmp2 = np.absolute(np.cross(AB, AP)) / mod
    tmp1 = np.where(AB_AP < 0., np.sqrt(np.sum(AP**2)), tmp2)
    return np.where(AB_BP > 0., np.sqrt(np.sum(BP**2)), tmp1)

d_to_line_segs = jax.jit(jax.vmap(d_to_line_seg, in_axes=(None, 0, 0), out_axes=0))


def sign_to_line_seg(P, O, A, B):
    ''' If P is inside the triangle OAB, return True, otherwise return False.
    '''
    OA = A - O
    OB = B - O
    OP = P - O
    AB = B - A
    AP = P - A
    OAxOB = np.cross(OA, OB)
    OAxOP = np.cross(OA, OP)
    OBxOP = np.cross(OB, OP)
    OAxAB = np.cross(OA, AB)
    ABxAP = np.cross(AB, AP)
    tmp2 = np.where(ABxAP * OAxAB < 0., False, True)
    tmp1 = np.where(OAxOB * OBxOP > 0., False, tmp2)
    return  np.where(OAxOB * OAxOP < 0., False, tmp1)

sign_to_line_segs = jax.jit(jax.vmap(sign_to_line_seg, in_axes=(None, None, 0, 0), out_axes=0))


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
    polygon_inertia_O = np.sum(triangle_inertias)
    polygon_inertia_G = polygon_inertia_O - np.sum(polygon_centroid**2)*polygon_area
    return polygon_area, polygon_inertia_G, polygon_centroid


def reference_to_physical(x1, x2, theta, ref_centroid, ref_points):
    rot = get_rot_mat(theta)
    points_wrt_centroid_initial = (ref_points - ref_centroid.reshape(1, -1)).reshape(ref_points.shape)
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

batch_eval_sdf = jax.vmap(eval_sdf, in_axes=(None, None, None, None, None, 0), out_axes=0)

grad_sdf = jax.grad(eval_sdf, argnums=(5))
batch_grad_sdf = jax.vmap(grad_sdf, in_axes=(None, None, None, None, None, 0), out_axes=0)


if __name__ == '__main__':
    pass
