import jax
import jax.numpy as np


def get_rot_mat(q):
    '''
    Transformation from quaternion to the corresponding rotation matrix.
    Reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    '''
    return np.array([[q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3], 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
                     [2*q[1]*q[2] + 2*q[0]*q[3], q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3], 2*q[2]*q[3] - 2*q[0]*q[1]],
                     [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]]])


get_rot_mats = jax.jit(jax.vmap(get_rot_mat, in_axes=0, out_axes=0))


def quat_mul(q, p):
    '''
    Quaternion multiplication.
    '''
    return np.array([q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3],
                     q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2],
                     q[0]*p[2] + q[2]*p[0] - q[1]*p[3] + q[3]*p[1],
                     q[0]*p[3] + q[3]*p[0] + q[1]*p[2] - q[2]*p[1]])

quats_mul = jax.jit(jax.vmap(quat_mul, in_axes=(0, 0), out_axes=0))