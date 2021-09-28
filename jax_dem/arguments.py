import numpy as onp
import jax
import jax.numpy as np
import argparse
import sys
import numpy as onp
import matplotlib.pyplot as plt
from jax.config import config

# config.update("jax_enable_x64", True)


# Set numpy printing format
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=10)
onp.random.seed(0)


# Manage arguments
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', help='Verbose for debug', action='store_true', default=True)
parser.add_argument('--gravity', type=float, default=9.8)
parser.add_argument('--env_bottom', type=float, default=10.)
parser.add_argument('--env_top', type=float, default=90.)
# parser.add_argument('--box_size', type=float, default=100.)
parser.add_argument('--dir', type=str, default='data')
parser.add_argument('--dim', type=int, default=3)
args = parser.parse_args()


# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

