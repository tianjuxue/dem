import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import vedo
from jax_dem.utils import get_rot_mats


def plot_energy(energy, file_path):
    plt.figure(num=10, figsize=(6, 6))
    plt.plot(20*np.arange(1, len(energy) + 1, 1), energy, marker='o',  markersize=2, linestyle="-", linewidth=1, color='blue')
    plt.xlabel("Time steps")
    plt.ylabel("Energy")
    plt.savefig(file_path)


def vedo_plot(case_name, radius, bottom, top, states=None):
    if states is None:
        states = np.load(f'data/numpy/vedo/states_{case_name}.npy')
 
    n_objects = states.shape[1]

    if hasattr(radius, "__len__"):
        radius = radius.reshape(-1)
    else:
        radius = np.array([radius] * n_objects)

    assert(radius.shape == (n_objects,))

    # This prob should be changed...
    if case_name == 'particles_in_drum' or case_name == 'objects_in_drum':
        world = vedo.Box(size=[bottom, top, bottom, top, bottom, top]).wireframe()
        vedo.show(world, axes=4, camera={'pos':[100, 50, 50], 'viewup':[0, 0, 1]}, interactive=0)
    elif case_name == 'billiards':
        world = vedo.Box(size=[40, 60, 40, 60, 10, 20]).wireframe()
        vedo.show(world, axes=4, camera={'pos':[50, 50, 60], 'viewup':[0, 1, 0]}, interactive=0)
    elif case_name == 'donuts':
        world = vedo.Box(size=[bottom, top, bottom, top, bottom, top]).wireframe()
        vedo.show(world, axes=4, viewup="z", interactive=0)
    else:
        raise ValueError()

    vd = vedo.Video(f"data/mp4/3d/{case_name}.mp4", fps=30)
    # Modify vd.options so that preview on Mac OS is enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    vd.options = "-b:v 8000k -pix_fmt yuv420p"

    for s in range(len(states)):
        x = states[s][:, 0:3]
        q = states[s][:, 3:7]
        initial_arrow = radius.reshape(-1, 1) * np.array([[0., 0., 1]])
        rot_matrices = get_rot_mats(q)
        endPoints = np.squeeze(rot_matrices @ initial_arrow[..., None], axis=-1) + x
        arrows = vedo.Arrows(startPoints=x, endPoints=endPoints, c="green")
        balls = vedo.Spheres(centers=x, r=radius, c="red", alpha=0.5)
        plotter = vedo.show(world, balls, arrows, resetcam=False)
        print(f"frame: {s} in {len(states) - 1}")
        vd.addFrame()

    vd.close() 
    # vedo.interactive().close()


