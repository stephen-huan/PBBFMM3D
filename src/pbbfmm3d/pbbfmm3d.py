from collections.abc import Callable
from pathlib import Path

import numpy as np

from .FMMCompute import (
    convert_to_numpy,
    convert_to_vecOfdouble,
    convert_to_vecOfvec3,
    vecOfdouble,
    vecOfvec3,
)
from .kernels import Kernel


def gram(kernel: Kernel, x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Gram matrix-vector product with the fast multipole method."""
    assert x.ndim == 2, f"Points must be a matrix, got {x.ndim} dimensions."
    n, d = x.shape
    assert d == 3, f"Dimension {d} must be 3."

    x_c = vecOfvec3()
    convert_to_vecOfvec3(x / kernel.length_scale, x_c)

    def matvec(y: np.ndarray) -> np.ndarray:
        """Matrix-vector product."""
        # seems to average values out for multiple columns
        assert y.ndim == 1, f"Got {y.ndim} columns, expected 1."
        assert y.shape[0] == n, f"Got a shape of {y.shape[0]}, expected {n}."
        assert kernel.tree is not None, "Call kernel.build() first."

        y_c = vecOfdouble()
        convert_to_vecOfdouble(y, y_c)
        out_c = vecOfdouble()
        out_c[:] = np.zeros(n)

        # HACK: ensure output directory for pbbfmm3d
        Path("output").mkdir(exist_ok=True, parents=True)
        # segfaults if the tree is not rebuild every iteration
        kernel.tree.buildFMMTree()
        kernel.compute(kernel.tree, x_c, x_c, y_c, 1, out_c)

        out = np.empty(n, dtype=np.float64)
        convert_to_numpy(out_c, out)
        return out

    return matvec


def cross_covariance(
    kernel: Kernel, x1: np.ndarray, x2: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Kernel matrix-vector product with the fast multipole method."""
    # implemented by reduction to fmm_gram since pbbfmm3d
    # can't seem to handle the case where x1 != x2.
    x = np.concatenate((x1, x2), axis=0)
    n = x1.shape[0]
    m = x2.shape[0]
    gram_matvec = gram(kernel, x)

    def matvec(y: np.ndarray) -> np.ndarray:
        """Matrix-vector product."""
        assert y.ndim == 1, f"Got {y.ndim} columns, expected 1."
        assert y.shape[0] == m, f"Got a shape of {y.shape[0]}, expected {m}."

        z = np.concatenate((np.zeros_like(y, shape=(n,)), y))
        return gram_matvec(z)[:n]

    return matvec
