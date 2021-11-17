import jax
import jax.numpy as np
import numpy as onp
from jax_dem.arguments import args
import trimesh
import vedo
import scipy.optimize as opt


def eval_num_ptcls(mesh, radius):
    '''
    Evaluates how many spherical particles roughly fit inside the mesh object.
    Namely, num_ptcls x vol_ptcl = vol_obj
    '''
    vol_ptcl = 4./3. * np.pi * radius**3
    vol_obj = mesh.volume
    num_ptcls = int(vol_obj/vol_ptcl)
    return num_ptcls


def sdfs_wrapper(mesh):
    '''
    The signed_distance function provided by trimesh is not differentiable, bad for optimization. 
    We define the jvp here so that sdfs_fn becomes a differentiable function.
    '''
    @jax.custom_jvp
    def sdfs_fn(points):
        signed_distances = trimesh.proximity.signed_distance(mesh, points)
        return -signed_distances
        
    @sdfs_fn.defjvp
    def sdf_jvp(primals, tangents):
        points, = primals
        tangents, = tangents
        closest_points, _, _ = trimesh.proximity.closest_point(mesh, points) 
        signed_distances = trimesh.proximity.signed_distance(mesh, points)
        directions = closest_points - points
        directions /= np.linalg.norm(directions, axis=-1)[:, None]
        tangents_out = np.sum(directions * tangents, axis=-1)
        return -signed_distances, np.where(-signed_distances < 0., tangents_out, -tangents_out)

    return sdfs_fn


def train_scipy(num_surface_ptcls, radius, mesh):
    key = jax.random.PRNGKey(0)
    points_ini = jax.random.uniform(key, shape=(num_surface_ptcls, 3), minval=-10., maxval=10.)
    sdfs_fn = sdfs_wrapper(mesh)

    def obj_fn(points_flat):
        points = points_flat.reshape(-1, 3)
        dists = np.sum((points[:, None, :] - points[None, :, :])**2, axis=-1)
        dists_loss = np.where(dists < (2*radius)**2, (2*radius)**2 - dists, 0.)
        diag_elements = np.diag_indices_from(dists_loss)
        dists_loss = dists_loss.at[diag_elements].set(0.)
        loss = np.sum(dists_loss)
        sdfs = sdfs_fn(points)
        reg = 1e3
        penalty = reg*np.sum((sdfs + radius)**2) 
        return loss + penalty

    obj_vals = []
    def objective(points_flat):
        obj_val = obj_fn(points_flat)
        obj_vals.append(obj_val)
        print(f"obj_val = {obj_val}")
        return obj_val

    def derivative(points_flat):
        der_val = jax.grad(obj_fn)(points_flat)
        return onp.array(der_val, order='F', dtype=onp.float64)

    options = {'maxiter': 300, 'disp': True}
    res = opt.minimize(fun=objective,
                       x0=points_ini,
                       method='L-BFGS-B',
                       jac=derivative,
                       callback=None,
                       options=options)
 
    points_opt = res.x.reshape(-1, 3)
    sdfs = sdfs_fn(points_opt)
    print(f"max sdfs = {np.max(sdfs)}, min sdfs = {np.min(sdfs)}")
    return points_opt


def vedo_plot(centers, radius, mesh):
    tetra = vedo.Mesh(mesh, alpha=0.5).lw(0.1)
    n_objects = centers.shape[0]
    if hasattr(radius, "__len__"):
        radius = radius.reshape(-1)
    else:
        radius = np.array([radius] * n_objects)

    assert(radius.shape == (n_objects,))

    bounds = mesh.bounding_box_oriented.bounds
    world = vedo.Box(size=bounds.T.reshape(-1)).wireframe()
    balls = vedo.Spheres(centers=centers, r=radius, c="red", alpha=1.)
    vedo.show(world, balls, tetra, axes=4, viewup="z", interactive=1)


def get_surface_ptcls(): 
    # mesh = trimesh.load(f'data/stl/tetra.stl')
    mesh = trimesh.load(f'data/stl/hollow_tetra.stl')
    mesh.vertices -= mesh.center_mass
    radius = 0.5
    num_ptcls = eval_num_ptcls(mesh, radius)
    print(f"num_ptcls = {num_ptcls} fits the object with radius = {radius}")
    sdfs_fn = sdfs_wrapper(mesh)
    num_surface_ptcls = 200
    points_opt = train_scipy(num_surface_ptcls, radius, mesh)
    vedo_plot(points_opt, radius, mesh)
    # np.save(f"data/numpy/convert/hollow_tetra.npy", points_opt)


def exp_trimesh():
    mesh = trimesh.load(f'data/stl/tetra.stl')
    # mesh = trimesh.load(f'data/stl/hollow_tetra.stl')
    mesh.vertices -= mesh.center_mass
    sdfs_fn = sdfs_wrapper(mesh)
    print(f"\nvertices = {mesh.vertices}")
    print(f"\nis_watertight = {mesh.is_watertight}")
    print(f"\nvolume = {mesh.volume}")
    print(f"\nmoment_inertia = \n{mesh.moment_inertia}")
    print(f"\ncenter_mass = {mesh.center_mass}")

    query_points = np.array([[0., 0., 0.], [100, 100, 100]])
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(mesh, query_points) 
    print(f"\nquery_points = \n{query_points}")
    print(f"\nclosest_points on object surface = \n{closest_points}")
    print(f"\nsdfs = {sdfs_fn(query_points)}" )
    print(f"\nsdf_grads = \n{jax.jacfwd(sdfs_fn)(query_points)}")


if __name__ == '__main__':
    get_surface_ptcls()
    # exp_trimesh()
