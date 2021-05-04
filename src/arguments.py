import argparse
import sys
import numpy as onp
import matplotlib.pyplot as plt
import jax

# Set numpy printing format
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=10)
onp.random.seed(0)


# Manage arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--verbose', help='Verbose for debug', action='store_true', default=True)
# args = parser.parse_args()


# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

