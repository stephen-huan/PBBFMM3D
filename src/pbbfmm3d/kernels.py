from collections.abc import Callable
from typing import Any

import numpy as np

from .FMMCompute import (
    ComputeMatern12,
    ComputeMatern32,
    ComputeMatern52,
    ComputeMaternInf,
    vecOfdouble,
    vecOfvec3,
)
from .FMMTree import (
    kernel_Matern12,
    kernel_Matern32,
    kernel_Matern52,
    kernel_MaternInf,
    vector3,
)

Tree = Any
BaseKernel = Callable[[float, int, int, float, int], Tree]
Compute = Callable[
    [Tree, vecOfvec3, vecOfvec3, vecOfdouble, int, vecOfdouble], None
]


class Kernel:
    """A kernel function."""

    base_kernel: BaseKernel
    compute: Compute
    length_scale: float = 1.0
    tree: Tree | None = None

    def __init__(
        self,
        base_kernel: BaseKernel,
        compute: Compute,
        /,
        length_scale: float = 1.0,
    ) -> None:
        self.base_kernel = base_kernel
        self.compute = compute
        self.length_scale = length_scale

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the kernel function."""
        assert x.ndim == 1 and y.ndim == 1, "Inputs are not points."
        assert x.size == 3 and y.size == 3, "Points are not 3d."
        assert self.tree is not None, "Call .build() first."

        xp = vector3()
        xp.x, xp.y, xp.z = x / self.length_scale
        yp = vector3()
        yp.x, yp.y, yp.z = y / self.length_scale
        return self.tree.EvaluateKernel(xp, yp)

    def build(
        self,
        L: float,
        tree_level: int,
        interpolation_order: int,
        eps: float = 1e-5,
        use_chebyshev: bool = False,
    ) -> None:
        """Build the tree."""
        tree = self.base_kernel(
            L / self.length_scale,
            tree_level,
            interpolation_order,
            eps,
            int(use_chebyshev),
        )
        tree.buildFMMTree()
        self.tree = tree


class Matern12(Kernel):
    """Matern kernel with smoothness nu = 1/2."""

    def __init__(self, length_scale: float = 1.0) -> None:
        self.base_kernel = kernel_Matern12
        self.compute = ComputeMatern12
        self.nu = 1 / 2
        self.length_scale = length_scale


class Matern32(Kernel):
    """Matern kernel with smoothness nu = 3/2."""

    def __init__(self, length_scale: float = 1.0) -> None:
        self.base_kernel = kernel_Matern32
        self.compute = ComputeMatern32
        self.nu = 3 / 2
        self.length_scale = length_scale


class Matern52(Kernel):
    """Matern kernel with smoothness nu = 5/2."""

    def __init__(self, length_scale: float = 1.0) -> None:
        self.base_kernel = kernel_Matern52
        self.compute = ComputeMatern52
        self.nu = 5 / 2
        self.length_scale = length_scale


class MaternInf(Kernel):
    """Matern kernel with infinite smoothness."""

    def __init__(self, length_scale: float = 1.0) -> None:
        self.base_kernel = kernel_MaternInf
        self.compute = ComputeMaternInf
        self.nu = np.inf
        self.length_scale = length_scale


# aliases
Exponential = Matern12
SquaredExponential = MaternInf


def from_sklearn(kernel: Any) -> Kernel:
    """Turn a scikit-learn kernel into a pbbfmm3d kernel."""
    import sklearn.gaussian_process.kernels as kernels

    params = kernel.get_params()
    if isinstance(kernel, kernels.Matern):
        fmm_kernels = {
            1 / 2: Matern12,
            3 / 2: Matern32,
            5 / 2: Matern52,
            np.inf: MaternInf,
        }
        length_scale, nu = params["length_scale"], params["nu"]
        assert nu in fmm_kernels, f"Smoothness {nu} is not implemented."
        return fmm_kernels[nu](length_scale=length_scale)
    elif isinstance(kernel, kernels.RBF):
        return SquaredExponential(length_scale=params["length_scale"])
    else:
        raise ValueError(f"Kernel {kernel} is not implemented.")


def from_gpjax(kernel: Any) -> Kernel:
    """Turn a gpjax kernel into a pbbfmm3d kernel."""
    from gpjax import kernels

    if isinstance(kernel, kernels.Matern12):
        return Matern12(length_scale=float(kernel.lengthscale.value))
    elif isinstance(kernel, kernels.Matern32):
        return Matern32(length_scale=float(kernel.lengthscale.value))
    elif isinstance(kernel, kernels.Matern52):
        return Matern52(length_scale=float(kernel.lengthscale.value))
    elif isinstance(kernel, kernels.RBF):
        return SquaredExponential(length_scale=float(kernel.lengthscale.value))
    else:
        raise ValueError(f"Kernel {kernel} is not implemented.")


__all__ = [
    "Exponential",
    "Matern12",
    "Matern32",
    "Matern52",
    "MaternInf",
    "SquaredExponential",
    "from_sklearn",
    "from_gpjax",
]
