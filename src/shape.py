import numpy as onp
import jax
import jax.numpy as np
import fenics as fe
import mshr
import unittest
import meshio
import numpy.testing as nptest
from .polyhedron import signed_tetrahedron_volume, tetrahedron_volume


class TestMesh(unittest.TestCase):

    def test_arbitrary_tetrahedron(self):
        '''
        Check if the orientations of surface cells are all good.
        '''
        total_vol = 0
        object_3D = generate_template_object(10)
        points = object_3D.get_points()
        connectivity = object_3D.get_connectivity()
        for conn in connectivity:
            tetra = np.concatenate([np.array([[0., 0., 0.]]), np.take(points, conn, axis=0)], axis=0)
            vol = signed_tetrahedron_volume(*tetra)
            assert vol > 0., "Orientation of the tetrahetron is wrong"
            total_vol += vol 
        print(f"V={total_vol}, should be around {4./3.*np.pi}")


def generate_template_object(resolution):
    '''
    The star-convex polyhedron consists of many tetrahedra. 
    This function generates the template for such a polyhedron with the initial shape close to a sphere.

    Parameter
    ---------
    resolution: larger value indicates more shape parameters

    Returns
    -------
    directions: numpy array with shape (num_points, dim)
    connectivity: numpy array with shape (num_cells, dim)
    '''
    sphere = mshr.Sphere(center=fe.Point(0, 0, 0), radius=1.)
    mesh = mshr.generate_mesh(sphere, resolution)
    file_mesh = fe.File('data/vtk/template/mesh.pvd')
    file_mesh << mesh
    points = onp.array(mesh.coordinates())
    connectivity = onp.array(mesh.cells())
    bmesh = fe.BoundaryMesh(mesh, "exterior")
    file_bmesh = fe.File('data/vtk/template/bmesh.pvd')
    file_bmesh << bmesh
    points = onp.array(bmesh.coordinates())
    connectivity = onp.array(bmesh.cells())
    directions = points / onp.linalg.norm(points, axis=1).reshape(-1, 1)
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

    print(f"Created template for a polyhedron, number of params={len(directions)}, number of surface triangles={len(connectivity)}")

    object_3D = ThreeDimObject(directions, connectivity)

    return object_3D


class ThreeDimObject:
    def __init__(self, directions, connectivity, params=None):
        '''
        Parameters
        ----------
        directions: numpy array with shape (num_points, dim)
        connectivity: numpy array with shape (num_cells, dim)
        '''
        self._directions = directions
        self._connectivity = connectivity
        if params is None:
            self.params = np.ones(len(directions))
        else:
            self.params = params


    def get_points(self, params=None):
        if params is None:
            params = self.params
        return params.reshape(-1, 1) * self._directions


    def get_connectivity(self):
        return self._connectivity


    def output_vtk(self):
        points = self.get_points()
        cells = [("triangle", self._connectivity)]
        mesh = meshio.Mesh(points,  cells)
        mesh.write("foo.vtk")


    def morph(self, shape_func):
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
            points = self.get_points(params)
            values = shape_func(points)
            params_upper = onp.where(values > 0., params, params_upper)
            params_lower = onp.where(values < 0., params, params_lower)
            params = (params_upper + params_lower) / 2.

        err = onp.max(onp.absolute(values))
        print(f"err={err}")
        self.params = np.array(params)


def get_cube():
    # cube_func = lambda x: np.linalg.norm(x, ord=1, axis=-1) - 1.
    cube_func = lambda x: np.max(np.absolute(x), axis=-1) - 1.
    object_3D = generate_template_object(20)
    object_3D.morph(cube_func)
    object_3D.output_vtk()


if __name__ == '__main__':
    # unittest.main()
    get_cube()
