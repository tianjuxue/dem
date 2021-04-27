import numpy as onp
import jax
import jax.numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from .data_generator import generate_radius_samples


@jax.jit
def d_to_line_seg(P, A, B):
    AB = B - A
    BP = P - B
    AP = P - A
    AB_BP = np.dot(AB, BP)
    AB_AP = np.dot(AB, AP)

    def f1(_):
        return np.sqrt(np.sum(BP**2))

    def f2(_):
        return np.sqrt(np.sum(AP**2)) 

    def f3(_):
        mod = np.sqrt(np.sum(AB**2))
        return np.absolute(np.cross(AB, AP)) / mod

    def f(_):
        return jax.lax.cond(AB_AP < 0., f2, f3, operand=None)

    return jax.lax.cond(AB_BP > 0., f1, f, operand=None)

d_to_line_segs = jax.vmap(d_to_line_seg, in_axes=(None, 0, 0), out_axes=0)


@jax.jit
def sign_to_line_seg(P, A, B):
    O = np.array([0., 0.])
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

    def f2(_):
        return jax.lax.cond(ABxAP * OAxAB < 0., lambda _:1., lambda _:-1., operand=None)

    def f1(_):
        return jax.lax.cond(OAxOB * OBxOP > 0., lambda _:1., f2, operand=None)

    return  jax.lax.cond(OAxOB * OAxOP < 0., lambda _:1., f1, operand=None)

sign_to_line_segs = jax.vmap(sign_to_line_seg, in_axes=(None, 0, 0), out_axes=0)


@jax.jit
def single_forward(radius_sample, input_point):
    radius_sample_rolled = np.roll(radius_sample, -1)
    angles = np.linspace(0, 2 * np.pi, len(radius_sample) + 1)[:-1]
    angles_rolled = np.roll(angles, -1)
    pointsA =  np.vstack([radius_sample*np.cos(angles), radius_sample*np.sin(angles)]).T
    pointsB = np.vstack([radius_sample_rolled*np.cos(angles_rolled), radius_sample_rolled*np.sin(angles_rolled)]).T
    result = np.min(d_to_line_segs(input_point, pointsA, pointsB)) * np.prod(sign_to_line_segs(input_point, pointsA, pointsB))
    return result

batch_forward = jax.vmap(single_forward, in_axes=(None, 0), out_axes=0)


def test(args):
    # radius_sample = np.array(np.load(os.path.join(args.dir, 'numpy/training/radius_samples.npy')))[0]
    radius_sample = generate_radius_samples(100, num_division=256)[2]
 
    P = np.array([0.00111, 0.00111])
    print(single_forward(radius_sample, P))
 
    xgrid = np.linspace(-args.domain_length, args.domain_length, 1000)
    x1, x2 = np.meshgrid(xgrid, xgrid)
    xx = np.vstack([x1.ravel(), x2.ravel()]).T
    out = np.reshape(batch_forward(radius_sample, xx), x1.shape)

    value, grads = jax.jit(jax.value_and_grad(lambda radius_sample, xx: np.sum(batch_forward(radius_sample, xx))))(radius_sample, P.reshape(1, -1))

    print(value)
    print(grads)

    plt.figure(num=0, figsize=(8, 8))
    plt.contourf(x1, x2, out, levels=50, cmap='seismic')
    plt.colorbar()
    contours = np.linspace(-1., 1., 11)
    plt.contour(x1, x2, out, contours, colors=['black']*len(contours))
    plt.contour(x1, x2, out, [0.], colors=['red'])
    plt.axis('equal')
    x_origin = np.array([[0., 0.]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()        
    parser.add_argument('--domain_length', type=float, default=2.)
    parser.add_argument('--dir', type=str, default='data')
    args = parser.parse_args()
    test(args)
    plt.show()
