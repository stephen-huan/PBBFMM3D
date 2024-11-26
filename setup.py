from os import environ as env
from pathlib import Path

from setuptools import Extension, setup

path = Path(__file__).parent

useMKL = "MKLROOT" in env
mkl = env.get("MKLROOT", None)
libboost = f":libboost_python{env['PYTHON_VERSION'].replace('.', '')}.so"
include_dirs = [
    env["BOOST_INC"],
    env["FFTW_INCLUDE"],
    env["PYTHON_INCLUDE"],
    path / "include",
    path / "python",
] + ([f"{mkl}/include"] if useMKL else [])
library_dirs = [env["BOOST_LIB"], env["FFTW_LIB"], env["PYTHON_LIB"]]
libraries = [
    "fftw3",
    "pthread",
    "m",
    "dl",
    f"python{env['PYTHON_VERSION']}",
    libboost,
    "boost_system",
    "boost_filesystem",
] + (["blas", "lapack"] if not useMKL else [])
extra_compile_args = ["-Wall", "-O3", "-fopenmp", "-fPIC"]
extra_link_args = [
    "-fopenmp",
    "-shared",
    "-Wl,--export-dynamic",
] + (
    [
        "-Wl,--start-group",
        f"{mkl}/lib/libmkl_intel_lp64.so",
        f"{mkl}/lib/libmkl_sequential.so",
        f"{mkl}/lib/libmkl_core.so",
        "-Wl,--end-group",
    ]
    if useMKL
    else []
)

setup(
    ext_modules=[
        Extension(
            name="pbbfmm3d.FMMCompute",
            sources=["src/FMMCompute.cpp", "src/H2_3D_Tree.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
        Extension(
            name="pbbfmm3d.FMMTree",
            sources=["src/FMMTree.cpp", "src/H2_3D_Tree.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]
)
