import time

import numpy as np
import sklearn.gaussian_process.kernels as kernels

from pbbfmm3d import gram
from pbbfmm3d.kernels import from_sklearn

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)


if __name__ == "__main__":
    D = 3  # dimension of points
    N = int(1e5)  # number of points
    n_cols = 10  # number of iterations

    X = rng.random((N, D))

    start = time.time()
    kernel = from_sklearn(kernels.Matern(length_scale=1, nu=5 / 2))
    kernel.init(L=1, tree_level=4, interpolation_order=5)
    kernel.build()
    matvec = gram(kernel, X)
    print(f"preprocessing took {time.time() - start:.3f}")
    for i in range(n_cols):
        y = rng.random((N,))
        inner_start = time.time()
        matvec(y)
        print(f"{i + 1:3}th iteration took {time.time() - inner_start:.3f}")
    print(f"took {time.time() - start:.3f} total")
