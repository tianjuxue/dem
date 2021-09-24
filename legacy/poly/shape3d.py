import numpy as onp
import jax
import jax.numpy as np
import fenics as fe
import mshr
import unittest
import meshio
import argparse
from scipy.spatial.transform import Rotation as R
from functools import partial
import numpy.testing as nptest
from .arguments import args
from .tetrahedron import signed_tetrahedron_volume, tetrahedron_volume, d_to_triangles, sign_to_tetrahedra, \
tetrahedra_centroids, tetra_inertia_tensors, tetrahedra_volumes, signed_tetrahedra_volumes
from .io import output_vtk_3D_field, output_vtk_3D_shape

dim = args.dim


class TestShape(unittest.TestCase):

    def test_arbitrary_tetrahedron(self):
        '''
        Check if the orientations of surface cells are all good.
        '''
        total_vol = 0
        object_3D = generate_template_object('sphere', 10)
        points = object_3D.get_vertices()
        connectivity = object_3D.get_connectivity()
        for conn in connectivity:
            tetra = np.concatenate([np.array([[0., 0., 0.]]), np.take(points, conn, axis=0)], axis=0)
            vol = signed_tetrahedron_volume(*tetra)
            assert vol > 0., "Orientation of the tetrahetron is wrong"
            total_vol += vol 
        print(f"V={total_vol}, should be around {4./3.*np.pi}")


    def test_sdf_regular_tetrahedron(self):
        '''
        Check if the signed distance function to a regular tetrahedron is correct.
        '''
        vertices = np.array([[1., 1., 1.], [1., -1., -1.], [-1., -1.,  1.], [-1., 1., -1.]])
        connectivity = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        vertices_oriented = np.take(vertices, connectivity.T, axis=0)
        origin = np.array([0., 0., 0.])
        scalar_func = partial(batch_eval_sdf_helper, vertices_oriented, origin)

        key = jax.random.PRNGKey(0)
        test_points = jax.random.uniform(key, shape=(1000, dim), minval=-3., maxval=3.)
        grads = batch_grad_sdf_helper(vertices_oriented, origin, test_points)
        grad = grad_sdf_helper(vertices_oriented, origin, origin)
 
        output_vtk_3D_field(args, scalar_func, f"data/vtk/3d/test/sdf_reg_tetra.vtk")
        output_vtk_3D_shape(vertices, connectivity, f"data/vtk/3d/test/shape_reg_tetra.vtk")


    def test_cube(self):
        '''
        Check if the signed distance function to a cube (made of tetrahedra) is correct.
        Check if the inertia tensor and volume of a cube is correct.
        '''
        vertices = np.array([[1., -1., -1.],
                             [1., 1., -1.],
                             [-1., 1., -1.],
                             [-1., -1., -1.],
                             [1., -1., 1.],
                             [1., 1., 1.],
                             [-1., 1., 1.],
                             [-1., -1., 1.]])
        connectivity = np.array([[1, 4, 0], [4, 1, 5], [7, 2, 3], [2, 7, 6], 
                                 [4, 3, 0], [3, 4, 7], [2, 5, 1], [5, 2, 6],
                                 [3, 1, 0], [1, 3, 2], [5, 7, 4], [7, 5, 6]])
        vertices_oriented = np.take(vertices, connectivity.T, axis=0)
        origin = np.array([0., 0., 0.])
        scalar_func = partial(batch_eval_sdf_helper, vertices_oriented, origin)
        output_vtk_3D_field(args, scalar_func, f"data/vtk/3d/test/sdf_cube.vtk")
        output_vtk_3D_shape(vertices, connectivity, f"data/vtk/3d/test/shape_cube.vtk")

        cube_func = lambda x: np.max(np.absolute(x), axis=-1) - 1.
        key = jax.random.PRNGKey(0)
        test_points = jax.random.uniform(key, shape=(1000, dim), minval=-1., maxval=1.)
        test_values = scalar_func(test_points)
        true_values = cube_func(test_points)
        nptest.assert_array_almost_equal(test_values, true_values, decimal=5)

        q = np.array([1., 0., 0., 0.])
        cube_intertia, cube_vol, ref_centroid = compute_inertia_tensor(np.ones(len(vertices)), vertices, connectivity, q)

        side = 2.
        cube_vol_exact = side**3
        cube_intertia_exact = 1./6. * cube_vol_exact * side**2 * np.eye(dim)

        nptest.assert_almost_equal(cube_vol, cube_vol_exact, decimal=5)
        nptest.assert_array_almost_equal(cube_intertia, cube_intertia_exact, decimal=5)
 

    def test_sdf_polyhedron(self):
        '''
        Check if the signed distance function to the template polyhedron makes sense.
        '''
        object_3D = generate_template_object('sphere', 10)
        vertices_oriented = object_3D.get_oriented_vertices()
        connectivity = object_3D.get_connectivity()
        vertices = object_3D.get_vertices()
        origin = np.array([0., 0., 0.])
        scalar_func = partial(batch_eval_sdf_helper, vertices_oriented, origin)
        output_vtk_3D_field(args, scalar_func, f"data/vtk/3d/test/sdf_cube.vtk")
        output_vtk_3D_shape(vertices, connectivity, f"data/vtk/3d/test/shape_poly.vtk")


    def test_morph_into_given_shape(self):
        '''
        Check if the morph_into_shape function works reasonably.
        '''
        cube_func = lambda x: np.max(np.absolute(x), axis=-1) - 1.
        object_3D = generate_template_object('sphere', 20)
        object_3D.morph_into_shape(cube_func)
        connectivity = object_3D.get_connectivity()
        vertices = object_3D.get_vertices()
        output_vtk_3D_shape(vertices, connectivity, f"data/vtk/3d/test/shape_morphed.vtk")


    def test_quaternion_rotation(self):
        '''
        Compare our version of quaternion-to-matrix function with scipy version
        '''
        q = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        b1 = r.as_matrix()
        b2 = get_rot_mat(q)
        nptest.assert_array_almost_equal(b1, b2, decimal=4)      


    def test_polyhedron_inertia_tensor_1(self):
        '''
        The inertia tensor should only depend on the orientation of the object.
        We compute the inertia tensor in two different frameworks and compare the results.
        '''
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, shape=(dim,))
        q = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
        object_3D = generate_template_object('sphere', 10)
        params = object_3D.params
        directions = object_3D.get_directions()
        connectivity = object_3D.get_connectivity()

        # First approach
        polyhedron_intertia_1, polyhedron_vol_1, ref_centroid = compute_inertia_tensor(params, directions, connectivity, q)

        # Second approach
        phy_vertices_oriented, phy_pointO = get_phy_vertices_oriented(params, directions, connectivity, ref_centroid, x, q)
        tetra_vols = tetrahedra_volumes(phy_pointO, *phy_vertices_oriented)
        polyhedron_vol_2 = np.sum(tetra_vols)
        tetra_centroids = tetrahedra_centroids(phy_pointO, *phy_vertices_oriented)
        phy_centroid = np.sum(tetra_vols.reshape(-1, 1)*tetra_centroids, axis=0) / polyhedron_vol_2
        polyhedron_intertia_2 = np.sum(tetra_inertia_tensors(phy_pointO, *phy_vertices_oriented, phy_centroid), axis=0)
 
        nptest.assert_almost_equal(polyhedron_vol_1, polyhedron_vol_2, decimal=4)
        nptest.assert_array_almost_equal(polyhedron_intertia_1, polyhedron_intertia_2, decimal=4)
       

    def test_polyhedron_inertia_tensor_2(self):
        '''
        The inertia tensor must follow the convection of tensor rotation, i.e., I' = R I R^-1, where R is the rotation tensor
        '''
        q1 = np.array([1., 0., 0., 0])
        q2 = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
        R = get_rot_mat(q2)

        cube_func = lambda x: np.max(np.absolute(x), axis=-1) - 1.
        object_3D = generate_template_object('sphere', 10)
        object_3D.morph_into_shape(cube_func)
        params = object_3D.params
        directions = object_3D.get_directions()
        connectivity = object_3D.get_connectivity()

        polyhedron_intertia_before, _, _ = compute_inertia_tensor(params, directions, connectivity, q1)
        polyhedron_intertia_after, _, _ = compute_inertia_tensor(params, directions, connectivity, q2) 

        nptest.assert_array_almost_equal(polyhedron_intertia_after, R @ polyhedron_intertia_before @ R.T, decimal=6)


def trapezoid_shape(offset=0.5):
    '''
    Custom initial shape like a trapezoid 3D shape.
    '''
    base = 2.
    width = base - offset
    length = base + offset
    height = base
    vertices = np.array([[length/2., -width/2., -height/2.],
                         [length/2., width/2., -height/2.],
                         [-length/2., width/2., -height/2.],
                         [-length/2., -width/2., -height/2.],
                         [width/2., -length/2., height/2.],
                         [width/2., length/2., height/2.],
                         [-width/2., length/2., height/2.],
                         [-width/2., -length/2., height/2.]])
    connectivity = np.array([[1, 4, 0], [4, 1, 5], [7, 2, 3], [2, 7, 6], 
                             [4, 3, 0], [3, 4, 7], [2, 5, 1], [5, 2, 6],
                             [3, 1, 0], [1, 3, 2], [5, 7, 4], [7, 5, 6]])

    output_vtk_3D_shape(vertices, connectivity, f'data/xml/3d/template/trapezoid/shape.xml')


def generate_template_object(name, resolution, seeds_level=0):
    '''
    The star-convex polyhedron consists of many tetrahedra. 
    This function generates the template for such a polyhedron.

    Parameter
    ---------
    name: string that defines the shape of the template
    resolution: larger value indicates more shape parameters
    seeds_level: related to the density of seeds on polyhedron surface 

    Returns
    -------
    directions: numpy array with shape (num_vertices, dim)
    connectivity: numpy array with shape (num_cells, dim)
    '''
    
    if name == "sphere":
        sphere = mshr.Sphere(center=fe.Point(0, 0, 0), radius=1.)
        mesh = mshr.generate_mesh(sphere, resolution)
        file_mesh = fe.File(f'data/vtk/3d/template/{name}/mesh.pvd')
        file_mesh << mesh
        points = onp.array(mesh.coordinates())
        connectivity = onp.array(mesh.cells())
        bmesh = fe.BoundaryMesh(mesh, "exterior")
    elif name == "trapezoid":
        trapezoid_shape(0.5)
        bmesh = fe.Mesh(f'data/xml/3d/template/trapezoid/shape.xml')
    else:
        raise ValueError('Unknown initial shape!')

    file_bmesh = fe.File(f'data/vtk/3d/template/{name}/bmesh.pvd')
    file_bmesh << bmesh

    bmesh_refined = bmesh
    for i in range(seeds_level):
        bmesh_refined = fe.refine(bmesh_refined)
    file_bmesh << bmesh_refined

    points = onp.array(bmesh.coordinates())
    connectivity = onp.array(bmesh.cells())
    params = onp.linalg.norm(points, axis=1)
    directions = points / params.reshape(-1, 1)
    # It's dumb that BoundaryMesh does not produce oriented cells as they claimed
    # https://fenicsproject.org/olddocs/dolfin/1.5.0/python/programmers-reference/cpp/mesh/BoundaryMesh.html
    # We need to manually flip directions for those cells not oriented
    # See https://fenicsproject.discourse.group/t/when-calculating-the-boundary-mesh-normal-vectors-the-sign-of-the-direction-is-incorrect/195
    bmesh.init_cell_orientations(fe.Expression(('x[0]', 'x[1]', 'x[2]'), degree=1))
    cell_orientations = bmesh.cell_orientations()
    for i in range(len(connectivity)):
        if cell_orientations[i] == 1:
            tmp = connectivity[i][0]
            connectivity[i][0] = connectivity[i][1]
            connectivity[i][1] = tmp
 
    object_3D = ThreeDimObject(directions, connectivity, params)
    object_3D.ref_seeds = np.array(bmesh_refined.coordinates())

    # vertices_oriented must follow the correct order, otherwise inertia tensor calculation will be wrong.
    vertices_oriented = object_3D.get_oriented_vertices()
    signed_volumes = signed_tetrahedra_volumes(np.array([0., 0., 0.]), *vertices_oriented)
    assert np.all(signed_volumes > 0), "Orientation of the vertices is wrong!"

    print(f"Created template for a polyhedron, number of params={len(directions)}, "
          f"number of surface triangles={len(connectivity)}, "
          f"number of seeds={len(object_3D.ref_seeds)}")

    return object_3D


class ThreeDimObject:
    def __init__(self, directions, connectivity, params=None):
        '''
        Parameters
        ----------
        directions: numpy array with shape (num_vertices, dim)
        connectivity: numpy array with shape (num_cells, dim)
        '''
        self._directions = directions
        self._connectivity = connectivity
        if params is None:
            self.params = np.ones(len(directions))
        else:
            self.params = params


    def get_connectivity(self):
        return self._connectivity


    def get_directions(self):
        return self._directions


    def get_vertices(self, params=None):
        if params is None:
            params = self.params
        return params.reshape(-1, 1) * self._directions


    def get_oriented_vertices(self):
        vertices = self.get_vertices()
        return np.take(vertices, self._connectivity.T, axis=0)
 

    def morph_into_shape(self, shape_func):
        '''
        Simple binary search to morph the object into the shape defined by shape_func

        Parameter
        ---------
        shape_func: a function that returns negative values if a point is inside the shape, otherwise positive
        '''
        params_upper = 1e1 * onp.ones_like(self.params)
        params_lower = 1e-1 * onp.ones_like(self.params)
        params = (params_upper + params_lower) / 2.
        steps = 100
        for step in range(steps):
            points = self.get_vertices(params)
            values = shape_func(points)
            params_upper = onp.where(values > 0., params, params_upper)
            params_lower = onp.where(values < 0., params, params_lower)
            params = (params_upper + params_lower) / 2.

        err = onp.max(onp.absolute(values))
        print(f"err={err}")
        self.params = np.array(params)


def get_rot_mat(q):
    '''
    Standard transformation from quaternion to the corresponding rotation matrix.
    Reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    '''
    return np.array([[q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3], 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
                     [2*q[1]*q[2] + 2*q[0]*q[3], q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3], 2*q[2]*q[3] - 2*q[0]*q[1]],
                     [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]]])


get_rot_mats = jax.jit(jax.vmap(get_rot_mat, in_axes=0, out_axes=0))


def quat_mul(q, p):
    '''
    Standard quaternion multiplication.
    '''
    return np.array([q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3],
                     q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2],
                     q[0]*p[2] + q[2]*p[0] - q[1]*p[3] + q[3]*p[1],
                     q[0]*p[3] + q[3]*p[0] + q[1]*p[2] - q[2]*p[1]])

quats_mul = jax.jit(jax.vmap(quat_mul, in_axes=(0, 0), out_axes=0))


def rotate_point(q, points):
    '''
    Rotate some input points according to the quaternion q
    '''
    rot = get_rot_mat(q)
    points_rotated = points @ rot.T
    return points_rotated


def reference_to_physical(x, q, ref_centroid, ref_points):
    '''
    Compute the physical positions of some input points given their reference positions.

    Parameters
    ----------
    x: physical position of centroid
    q: quaternion (determines the rotation of the object)
    ref_centroid: reference position of centroid
    ref_points: reference positions of input points; can be of shape (dim,) or (batch, dim)

    Returns
    -------
    phy_points: physical positions of input points

    '''
    points_wrt_centroid_initial = (ref_points - ref_centroid.reshape(1, -1)).reshape(ref_points.shape)
    points_wrt_centroid = rotate_point(q, points_wrt_centroid_initial)
    phy_centroid = x
    phy_points = points_wrt_centroid + phy_centroid
    return phy_points


batch_reference_to_physical = jax.jit(jax.vmap(reference_to_physical, in_axes=(0, 0, None, None), out_axes=0))



#TODO: How to make seeds work reasonably well?
def get_ref_seeds(params, directions, connectivity):
    vertices = params.reshape(-1, 1) * directions
    return vertices


def get_phy_seeds(params, directions, connectivity, ref_centroid, x, q):
    ref_seeds = get_ref_seeds(params, directions, connectivity)
    phy_seeds = reference_to_physical(x, q, ref_centroid, ref_seeds)
    return phy_seeds


def get_ref_vertices_oriented(params, directions, connectivity):
    '''
    Returns
    -------
    ref_vertices_oriented: numpy array of shape (dim, num_params, dim)
    ref_pointO: numpy array of shape (dim,)
    '''
    ref_pointO = np.array([0., 0., 0.])
    vertices = params.reshape(-1, 1) * directions
    ref_vertices_oriented = np.take(vertices, connectivity.T, axis=0)
    return ref_vertices_oriented, ref_pointO


def get_phy_vertices_oriented(params, directions, connectivity, ref_centroid, x, q):
    '''
    Returns
    -------
    phy_vertices_oriented: numpy array of shape (dim, num_params, dim)
    phy_pointO: numpy array of shape (dim,)
    '''
    ref_vertices_oriented, ref_pointO = get_ref_vertices_oriented(params, directions, connectivity)
    phy_pointO = reference_to_physical(x, q, ref_centroid, ref_pointO)
    phy_vertices_oriented = reference_to_physical(x, q, ref_centroid, 
        ref_vertices_oriented.reshape(dim*len(connectivity), dim)).reshape(dim, len(connectivity), dim)

    return phy_vertices_oriented, phy_pointO


def eval_sign_helper(vertices_oriented, origin, point):
    '''
    Sign of a point to a polyhedron defined vertices_oriented and origin
    '''
    sign = np.where(np.any(sign_to_tetrahedra(point, origin, *vertices_oriented)), -1., 1.)    
    return sign


def eval_sdf_helper(vertices_oriented, origin, point):
    '''
    Signed distance function of a point to a polyhedron defined vertices_oriented and origin
    '''
    sign = eval_sign_helper(vertices_oriented, origin, point)
    sdf = np.min(d_to_triangles(point, *vertices_oriented)) * sign
    return sdf

batch_eval_sdf_helper = jax.jit(jax.vmap(eval_sdf_helper, in_axes=(None, None, 0), out_axes=0))

grad_sdf_helper = jax.grad(eval_sdf_helper, argnums=(-1))
batch_grad_sdf_helper = jax.jit(jax.vmap(grad_sdf_helper, in_axes=(None, None, 0), out_axes=0))


def eval_sign(params, directions, connectivity, ref_centroid, x, q, phy_point):
    '''
    Evaluate the sign of a physical point to a polyhedron.
    The polyhedron is defined by its shape (params, directions, connectivity) and 
    its translational and rotational position (ref_centroid, x, q)
    '''
    phy_vertices_oriented, phy_pointO = get_phy_vertices_oriented(params, directions, connectivity, ref_centroid, x, q)
    sign = eval_sign_helper(phy_vertices_oriented, phy_pointO, phy_point)
    return sign

batch_eval_sign = jax.jit(jax.vmap(eval_sign, in_axes=(None,)*6 + (0,), out_axes=0))


def eval_sdf(params, directions, connectivity, ref_centroid, x, q, phy_point):
    '''
    Evaluate the signed distance function of a physical point to a polyhedron.
    The polyhedron is defined by its shape (params, directions, connectivity) and 
    its translational and rotational position (ref_centroid, x, q)
    '''
    phy_vertices_oriented, phy_pointO = get_phy_vertices_oriented(params, directions, connectivity, ref_centroid, x, q)
    sdf = eval_sdf_helper(phy_vertices_oriented, phy_pointO, phy_point)
    return sdf

batch_eval_sdf = jax.jit(jax.vmap(eval_sdf, in_axes=(None,)*6 + (0,), out_axes=0))

grad_sdf = jax.grad(eval_sdf, argnums=(6))
batch_grad_sdf = jax.jit(jax.vmap(grad_sdf, in_axes=(None,)*6 + (0,), out_axes=0))


def compute_inertia_tensor(params, directions, connectivity, q):
    '''
    Compute intertia tensor for the polyhedron defined by its shape (params, directions, connectivity) and 
    its rotational position q
    '''
    ref_vertices_oriented, ref_pointO = get_ref_vertices_oriented(params, directions, connectivity)
    rotated_vertices_oriented = rotate_point(q, ref_vertices_oriented.reshape(dim*len(connectivity), dim)).reshape(dim, len(connectivity), dim)
    polyhedron_vol, ref_centroid = compute_volume_and_ref_centroid(params, directions, connectivity)
    rotated_centroid = rotate_point(q, ref_centroid)
    polyhedron_intertia = np.sum(tetra_inertia_tensors(ref_pointO, *rotated_vertices_oriented, rotated_centroid), axis=0)
    return polyhedron_intertia, polyhedron_vol, ref_centroid

compute_inertia_tensors = jax.jit(jax.vmap(compute_inertia_tensor, in_axes=(None, None, None, 0), out_axes=(0, None, None)))


def compute_volume_and_ref_centroid(params, directions, connectivity):
    '''
    Returns
    -------
    polyhedron_vol: volume of the polyhedron
    ref_centroid: the centroid position in the reference coordinate system
    '''
    ref_vertices_oriented, ref_pointO = get_ref_vertices_oriented(params, directions, connectivity)
    tetra_vols = tetrahedra_volumes(ref_pointO, *ref_vertices_oriented)
    polyhedron_vol = np.sum(tetra_vols)
    tetra_centroids = tetrahedra_centroids(ref_pointO, *ref_vertices_oriented)
    ref_centroid = np.sum(tetra_vols.reshape(-1, 1)*tetra_centroids, axis=0) / polyhedron_vol
    return polyhedron_vol, ref_centroid


if __name__ == '__main__':
    unittest.main()
