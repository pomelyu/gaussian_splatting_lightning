import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

PROJECT_NAME = "gs_kernels"

def get_extensions():
    use_debug = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("CUDA", "1") == "1"
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None

    if use_debug:
        print("Compiling in debug mode")

    extension = CUDAExtension if use_cuda else CppExtension
    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not use_debug else "-O0",
            # Define Py_LIMITED_API with min version 3.9 to expose only the stable
            # limited API subset from torch 
            "-DPy_LIMITED_API=0x03090000",
        ],
        "nvcc": [
            "-O3" if not use_debug else "-O0",
            "-DUSE_CUDA=1",
        ]
    }

    if use_debug:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    pwd = os.path.dirname(os.path.curdir)
    sources = list(glob.glob(os.path.join(pwd, PROJECT_NAME, "csrc", "*.cpp")))
    if use_cuda:
        sources += list(glob.glob(os.path.join(pwd, PROJECT_NAME, "csrc", "cuda", "*.cu")))

    ext_modules = [
        extension(
            f"{PROJECT_NAME}._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            # Build 1 wheel across multiple Python versions
            py_limited_api=True,
        )
    ]

    return ext_modules


setup(
    name=PROJECT_NAME,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension,
    },
    options={
        "bdist_wheel": {
            "py_limited_api": "cp39",   # 3.9 is minimum supported Python version
        }
    },
)
