import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
import argparse
import json
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dir = 'data'
    now = '1618998629'
    with open(os.path.join(args.dir, f'numpy/model/{now}/args.txt'), 'r') as f:
        args.__dict__ = json.load(f)
    print(args.__dict__)

    epoch = 9
    network_params, latent_params = np.load(os.path.join(args.dir, f'numpy/model/{now}/params_at_epoch_{epoch}.npy'), allow_pickle=True)

    print(latent_params)