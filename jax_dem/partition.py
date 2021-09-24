import numpy as onp
import dataclasses
import jax
from functools import reduce, partial
from typing import Any, Callable, Optional, Dict, Tuple, Generator, Union
from operator import mul
import jax
import jax.numpy as np
from jax import lax
from jax import ops
from jax.api import jit, vmap, eval_shape
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
import jax.numpy as np


Array = np.ndarray

i32 = np.int32
i64 = np.int64

f32 = np.float32
f64 = np.float64


def _cell_dimensions(spatial_dimension: int,
                     box_size: Array,
                     minimum_cell_size: float) -> Tuple[Array, Array, Array, int]:
    """Compute the number of cells-per-side and total number of cells in a box."""
    if isinstance(box_size, int) or isinstance(box_size, float):
        box_size = float(box_size)

    # NOTE(schsam): Should we auto-cast based on box_size? I can't imagine a case
    # in which the box_size would not be accurately represented by an f32.
    if (isinstance(box_size, onp.ndarray) and (box_size.dtype == np.int32 or box_size.dtype == np.int64)):
        box_size = float(box_size)


    cells_per_side = onp.floor(box_size / minimum_cell_size)
    cell_size = box_size / cells_per_side
    cells_per_side = onp.array(cells_per_side, dtype=np.int64)

    if isinstance(box_size, onp.ndarray):
        if box_size.ndim == 1 or box_size.ndim == 2:
            assert box_size.size == spatial_dimension
            flat_cells_per_side = onp.reshape(cells_per_side, (-1,))
            for cells in flat_cells_per_side:
                if cells < 3:
                    raise ValueError(('Box must be at least 3x the size of the grid spacing in each dimension.'))
            cell_count = reduce(mul, flat_cells_per_side, 1)
        elif box_size.ndim == 0:
            cell_count = cells_per_side ** spatial_dimension
        else:
            raise ValueError('Box must either be a scalar or a vector.')
    else:
        cell_count = cells_per_side ** spatial_dimension

    return box_size, cell_size, cells_per_side, int(cell_count)



def _compute_hash_constants(spatial_dimension: int,
                            cells_per_side: Array) -> Array:

    if cells_per_side.size == 1:
        return np.array([[cells_per_side ** d for d in range(spatial_dimension)]], dtype=np.int64)[::-1]
    elif cells_per_side.size == spatial_dimension:
        one = np.array([1], dtype=np.int32)
        cells_per_side = np.concatenate((one, cells_per_side[:-1]))
        return np.array(np.cumprod(cells_per_side), dtype=np.int64)[::-1]
    else:
        raise ValueError()

 
def _unflatten_cell_buffer(arr: Array,
                           cells_per_side: Array,
                           dim: int) -> Array:
    if (isinstance(cells_per_side, int) or
        isinstance(cells_per_side, float) or
        (isinstance(cells_per_side, np.ndarray) and not cells_per_side.shape)):
        cells_per_side = (int(cells_per_side),) * dim
    elif isinstance(cells_per_side, np.ndarray) and len(cells_per_side.shape) == 1:
        cells_per_side = tuple([int(x) for x in cells_per_side[::-1]])
    elif isinstance(cells_per_side, np.ndarray) and len(cells_per_side.shape) == 2:
        cells_per_side = tuple([int(x) for x in cells_per_side[0][::-1]])
    else:
        raise ValueError() # TODO
    return np.reshape(arr, cells_per_side + (-1,) + arr.shape[1:])

 
def cell_fn(R, box_size, minimum_cell_size, cell_capacity):
    '''Simplified from JAX-MD
    cells_per_side: (dim,)
    cell_id: (*cells_per_side, capacity, 1)
    indices: (n_objects, dim)
    '''    
    N = R.shape[0] 
    dim = R.shape[1]
 
    _, cell_size, cells_per_side, cell_count = _cell_dimensions(dim, box_size, minimum_cell_size)
    hash_multipliers = _compute_hash_constants(dim, cells_per_side)

    # Create cell list data.
    particle_id = lax.iota(np.int64, N)
    # NOTE(schsam): We use the convention that particles that are successfully,
    # copied have their true id whereas particles empty slots have id = N.
    # Then when we copy data back from the grid, copy it to an array of shape
    # [N + 1, output_dimension] and then truncate it to an array of shape
    # [N, output_dimension] which ignores the empty slots.

    cell_id = N * np.ones((cell_count * cell_capacity, 1), dtype=i32)
    indices = np.array(R / cell_size, dtype=i32)
    hashes = np.sum(indices * hash_multipliers, axis=1)

    # Copy the particle data into the grid. Here we use a trick to allow us to
    # copy into all cells simultaneously using a single lax.scatter call. To do
    # this we first sort particles by their cell hash. We then assign each
    # particle to have a cell id = hash * cell_capacity + grid_id where grid_id
    # is a flat list that repeats 0, .., cell_capacity. So long as there are
    # fewer than cell_capacity particles per cell, each particle is guarenteed
    # to get a cell id that is unique.
    sort_map = np.argsort(hashes)
    sorted_hash = hashes[sort_map]
    sorted_id = particle_id[sort_map]

    sorted_cell_id = np.mod(lax.iota(np.int64, N), cell_capacity)
    sorted_cell_id = sorted_hash * cell_capacity + sorted_cell_id

    sorted_id = np.reshape(sorted_id, (N, 1))
    cell_id = ops.index_update(cell_id, sorted_cell_id, sorted_id)
    cell_id = _unflatten_cell_buffer(cell_id, cells_per_side, dim)

    return cell_id, indices


def indices_1_to_27(indices):
    '''
    indices: (n_objects, dim)
    offsets: (dim, 27)
    expanded_indices: (n_objects, dim, 27)
    '''
    offset = np.arange(-1, 2, 1, dtype=np.int32)
    offsets = np.stack(np.meshgrid(*([offset]*3), indexing='ij')).reshape(3, -1)
    expanded_indices = indices[:, :, None] + offsets[None, :, :]
    return np.transpose(expanded_indices, axes=(1, 0, 2))
