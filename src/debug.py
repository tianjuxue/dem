import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from functools import partial
 
dim = 3 

onp.set_printoptions(precision=15)


def norm(x):
  x = np.sum(x**2)
  safe_x = np.where(x > 0., x, 0.)
  return np.sqrt(safe_x)


def d_to_line_seg(P, A, B):
    '''
    Distance of a point P to a line segment AB
    '''
    BA = B - A
    PB = P - B
    PA = P - A
    tmp1 = np.where(np.dot(BA, PA) < 0., norm(PA), norm(np.cross(BA, PA)) / norm(BA))
    return np.where(np.dot(BA, PB) > 0., norm(PB), tmp1)


def d_to_triangle(P, P1, P2, P3):
    '''
    Distance of a point P to a triangle (P1, P2, P3)
    Reference: https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
    '''
    u = P2 - P1
    v = P3 - P1
    w = P3 - P2
    n = np.cross(u, v)
    r = P - P1  
    s = P - P2
    t = P - P3

    n_square = np.sum(n*n)
    c3 = np.dot(np.cross(u, r), n) / n_square
    c2 = np.dot(np.cross(r, v), n) / n_square
    c1 = 1 - c3 - c2

    d1 = d_to_line_seg(P, P2, P3)
    d2 = d_to_line_seg(P, P1, P3)
    d3 = d_to_line_seg(P, P1, P2)
    d = np.min(np.array([d1, d2, d3]))

    tmp2 = np.where(c3 < 0., d, norm(P - (c1*P1 + c2*P2 + c3*P3)))
    tmp1 = np.where(c2 < 0., d, tmp2)
    return np.where(c1 < 0., d, tmp1)

grad_d_to_triangle = jax.grad(d_to_triangle)


def main():
    seed = np.array([5.8666096, 14.017725, 2.4854915])
    trias =np.array([[6.022082, 13.59783, 2.4969475], 
                     [5.7428236, 13.699588, 2.5707238],
                     [5.8992176, 13.94871, 2.485139]])

    value = d_to_triangle(seed, *trias)
    grad = grad_d_to_triangle(seed, *trias)

    print(value)
    print(grad)

    seed_ = seed + 1e-5

    value = d_to_triangle(seed_, *trias)
    grad = grad_d_to_triangle(seed_, *trias)

    print(value)
    print(grad)
 

if __name__ == '__main__':
    main()
