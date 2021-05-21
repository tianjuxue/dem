import numpy as onp
import jax
import jax.numpy as np
from jax.numpy.linalg import norm
import unittest
import numpy.testing as nptest


dim = 3

class TestTetrahedron(unittest.TestCase):

    def test_arbitrary_tetrahedron(self):
        '''
        Tetrahedron (O, D, E, F) has known centroid at G and inertia tensor I_G w.r.t. G
        Reference: doi.org/10.3844/jmssp.2005.8.11 (looks like Eq. 2 has typos)
        '''
        O = np.array([8.33220,  -11.86875, 0.93355])
        D = np.array([0.75523,   5.00000,  16.37072])
        E = np.array([52.61236,  5.00000, -5.38580])
        F = np.array([2.00000,   5.00000,  3.00000])
        G = np.array([15.92492,  0.78281,  3.72962])
        I_G = np.array([[ 43520.33257, -11996.20119,   46343.16662],
                        [-11996.20119,  194711.28938, -4417.66150],
                        [ 46343.16662, -4417.66150,    191168.76173]])

        nptest.assert_array_almost_equal(tetrahedron_center(O, D, E, F), G, decimal=3)
        nptest.assert_array_almost_equal(inertia_tensor(O, D, E, F, G), I_G, decimal=1)
 
    def test_regular_tetrahedron(self):
        '''
        Tetrahedron (O, D, E, F) is regular and has an analytical formula for inertia tensor
        Reference: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        '''
        O = np.array([ 1.,   1.,  1.])
        D = np.array([ 1.,  -1., -1.])
        E = np.array([-1.,  -1.,  1.])
        F = np.array([-1.,   1., -1.])
        G = np.array([ 0.,   0.,  0.])
       
        s = np.linalg.norm(D - O)
        vol = tetrahedron_volume(O, D, E, F)
        I_G = 1./20. * vol * s**2 * np.eye(dim)

        nptest.assert_array_almost_equal(tetrahedron_center(O, D, E, F), G, decimal=5)
        nptest.assert_array_almost_equal(inertia_tensor(O, D, E, F, G), I_G, decimal=5)


def signed_tetrahedron_volume(O, D, E, F):
    '''
    Signed volume of a tetrahedron with vertices (O, D, E, F)
    '''
    DO = D - O
    ED = E - D
    FD = F - D
    return np.dot(DO, np.cross(ED, FD)) / 6.


def tetrahedron_volume(O, D, E, F):
    '''
    Volume of a tetrahedron with vertices (O, D, E, F)
    ''' 
    return np.absolute(signed_tetrahedron_volume(O, D, E, F))


def tetrahedron_center(O, D, E, F):
    '''
    Mass center of a tetrahedron with vertices (O, D, E, F) 
    '''
    return (O + D + E + F) / 4.

 
def inertia_tensor(O, D, E, F, P):
    '''
    Inertia tensor of a tetrahedron with vertices (O, D, E, F) w.r.t an arbitrary point point P
    Use parallel axis theorem (see "Tensor generalization" on Wikipeida page "Parallel axis theorem")
    '''
    vol = tetrahedron_volume(O, D, E, F)
    center = tetrahedron_center(O, D, E, F)
    r_P = P - center
    r_O = O - center
    tmp_P = vol * (np.dot(r_P, r_P) * np.eye(dim) - np.outer(r_P, r_P))
    tmp_O = vol * (np.dot(r_O, r_O) * np.eye(dim) - np.outer(r_O, r_O))
    I_O = inertia_tensor_helper(O, D, E, F)
    I_P = I_O - tmp_O + tmp_P
    return I_P


def inertia_tensor_helper(O, D, E, F):
    '''
    Inertia tensor of a tetrahedron with vertices (O, D, E, F) w.r.t point O
    Reference: https://doi.org/10.1006/icar.1996.0243
    Unfortunately, it looks like the order of the vertices does matter when using this method. 
    '''    
    DO = D - O
    EO = E - O
    FO = F - O
    vol = signed_tetrahedron_volume(O, D, E, F)
    assert vol > 0, "Consider changing the order of vertices!"
    P = []
    for i in range(dim):
        P.append([])
        for j in range(dim):
            tmp = 2 * (DO[i] * DO[j] + EO[i] * EO[j] + FO[i] * FO[j]) + DO[i] * EO[j] + DO[j] * EO[i] + \
                  DO[i] * FO[j] +  DO[j] * FO[i] + EO[i] * FO[j] +  EO[j] * FO[i]
            P[i].append(tmp)
    I = [[ P[1][1] + P[2][2], -P[0][1],           -P[0][2]], 
         [-P[1][0],            P[0][0] + P[2][2], -P[1][2]], 
         [-P[2][0],           -P[2][1],            P[0][0] + P[1][1]]]
    return vol / 20. * np.array(I)



#TODO: Unit test of these functions


def d_to_triangle(P, P1, P2, P3):
    '''
    Distance of a point P to a triangle (P1, P2, P3)
    References: 
    https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
    https://www.geogebra.org/m/ZuvmPjmy
    '''
    u = P2 - P1
    v = P3 - P1
    w = P3 - P2
    n = np.cross(u, v)
    r = P - P1  
    s = P - P2
    t = P - P3

    n_square = np.sum(n*n)
    c1 = np.dot(np.cross(u, r), n) / n_square
    c2 = np.dot(np.cross(r, v), n) / n_square
    c3 = 1 - c1 - c2

    tmp5 = np.where(c3 < 0., norm(np.cross(r, u))/norm(u), norm(P - (c1*P1 + c2*P2 + c3*P3)))
    tmp4 = np.where(c3 < 0., norm(r), norm(np.cross(r, v))/norm(v))
    tmp3 = np.where(c2 < 0., tmp4, tmp5)
    tmp2 = np.where(c3 < 0., norm(s), norm(np.cross(s, w))/norm(w))
    tmp1 = np.where(c2 < 0., norm(t), tmp2)
    return np.where(c1 < 0., tmp1, tmp3)

d_to_triangles = jax.jit(jax.vmap(d_to_triangle, in_axes=(None, 0, 0, 0), out_axes=0))


def sign_to_tetrahedron(P, O, D, E, F):
    ''' 
    If P is inside the tetrahedron (O, D, E, F), return True, otherwise return False.
    '''
    DO = D - O
    EO = E - O
    FO = F - O
    ED = E - D
    FD = F - D
    OD = O - D 
    tmp3 = np.where(np.dot(np.cross(ED, FD), OD)*np.dot(np.cross(ED, FD), PD) < 0., False, True)
    tmp2 = np.where(np.dot(np.cross(EO, FO), DO)*np.dot(np.cross(EO, FO), PO) < 0., False, tmp3)
    tmp1 = np.where(np.dot(np.cross(DO, FO), EO)*np.dot(np.cross(DO, FO), PO) < 0., False, tmp2)
    return np.where(np.dot(np.cross(DO, EO), FO)*np.dot(np.cross(DO, EO), PO) < 0., False, tmp1)

sign_to_tetrahedra = jax.jit(jax.vmap(sign_to_tetrahedron, in_axes=(None, None, 0, 0, 0), out_axes=0))


if __name__ == '__main__':
    unittest.main()
