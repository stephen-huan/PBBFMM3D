import numpy as np
import sklearn.gaussian_process.kernels as kernels
from gpjax import kernels as jax_kernels

from pbbfmm3d import cross_covariance, gram
from pbbfmm3d.kernels import from_gpjax, from_sklearn

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)


if __name__ == "__main__":
    D = 3  # dimension of points
    N = 50  # number of points
    M = 25  # number of points

    smoothnesses = np.array([1 / 2, 3 / 2, 5 / 2, np.inf])
    length_scales = 10.0 ** np.arange(-2, 3)

    X1 = rng.random((N, D))
    X2 = rng.random((M, D))
    y1 = rng.random((N,))
    y2 = rng.random((M,))

    for length_scale in length_scales:
        sklean_kernels = [
            kernels.Matern(length_scale=length_scale, nu=nu)
            for nu in smoothnesses
        ] + [kernels.RBF(length_scale=length_scale)]
        gpjax_kernels = [
            jax_kernels.Matern12(lengthscale=length_scale),
            jax_kernels.Matern32(lengthscale=length_scale),
            jax_kernels.Matern52(lengthscale=length_scale),
            jax_kernels.RBF(lengthscale=length_scale),
            jax_kernels.RBF(lengthscale=length_scale),
        ]
        for kernel, kernel2 in zip(sklean_kernels, gpjax_kernels):
            theta_11: np.ndarray = kernel(X1, X1)  # type: ignore
            theta_12: np.ndarray = kernel(X1, X2)  # type: ignore
            theta_21: np.ndarray = kernel(X2, X1)  # type: ignore
            theta_22: np.ndarray = kernel(X2, X2)  # type: ignore

            fmm_kernel = from_sklearn(kernel)
            fmm_kernel2 = from_gpjax(kernel2)
            fmm_kernel.init(L=1, tree_level=5, interpolation_order=5)
            fmm_kernel2.init(L=1, tree_level=5, interpolation_order=5)
            for i in range(N):
                for j in range(N):
                    assert np.isclose(
                        fmm_kernel(X1[i], X1[j]), theta_11[i, j]
                    ), f"fmm entries wrong for {kernel}."
                    assert np.isclose(
                        fmm_kernel2(X1[i], X1[j]), theta_11[i, j]
                    ), f"fmm entries wrong for {kernel2}."
            assert np.allclose(
                gram(fmm_kernel, X1)(y1), theta_11 @ y1, rtol=1e-3
            ), f"fmm gram 1 wrong for {kernel}."
            assert np.allclose(
                gram(fmm_kernel, X2)(y2), theta_22 @ y2, rtol=1e-3
            ), f"fmm gram 2 wrong for {kernel}."
            assert np.allclose(
                cross_covariance(fmm_kernel, X1, X2)(y2),
                theta_12 @ y2,
                rtol=1e-2,
                atol=1e-3,
            ), f"fmm cross 12 wrong for {kernel}."
            assert np.allclose(
                cross_covariance(fmm_kernel, X2, X1)(y1),
                theta_21 @ y1,
                rtol=1e-2,
                atol=1e-4,
            ), f"fmm cross 21 wrong for {kernel}."
