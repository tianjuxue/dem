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


get_rot_mats = jax.vmap(get_rot_mat, in_axes=0, out_axes=0)


def quat_mul(q, p):
    '''
    Quaternion multiplication.
    '''
    return np.array([q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3],
                     q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2],
                     q[0]*p[2] + q[2]*p[0] - q[1]*p[3] + q[3]*p[1],
                     q[0]*p[3] + q[3]*p[0] + q[1]*p[2] - q[2]*p[1]])

quats_mul = jax.vmap(quat_mul, in_axes=(0, 0), out_axes=0)


def rotate_vector(q, vectors):
    '''
    Rotate rank 1 tensor (i.e., a vector) by quaternion q
    '''
    rot = get_rot_mat(q)
    vectors_rotated = vectors @ rot.T
    return vectors_rotated

rotate_vector_batch = jax.vmap(rotate_vector, in_axes=(0, 0), out_axes=0)


def rotate_tensor(q, tensors):
    '''
    Rotate rank 2 tensor by quaternion q
    '''
    rot = get_rot_mat(q)
    tensors_rotated = rot @ tensors @ rot.T
    return tensors_rotated


def norm(x):
    '''
    Safe norm to avoid jax.grad yielding np.nan at singular point
    '''
    x = np.sum(x**2, axis=-1)
    safe_x = np.where(x > 0., x, 0.)
    return np.sqrt(safe_x)


def get_unit_vectors(vectors):
    '''
    Get normalized vectors
    '''
    norms = norm(vectors)
    norms_reg = np.where(norms == 0., 1., norms)
    unit_vectors = vectors / norms_reg[..., None]
    return norms, unit_vectors
