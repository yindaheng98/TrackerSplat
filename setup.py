#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup, find_packages, find_namespace_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

packages = ['instantsplatstream'] + ["instantsplatstream." + package for package in find_packages(where="instantsplatstream")]
packages_dot = ['dot'] + ["dot." + package for package in find_namespace_packages(where="submodules/dot/dot")]
featurefusion_root = "submodules/featurefusion"
motionfusion_root = "submodules/motionfusion"
rasterizor_sources = [
    "cuda_rasterizer/rasterizer_impl.cu",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/backward.cu",
    "rasterize_points.cu",
    "ext.cpp"]
rasterizor_packages = {
    'instantsplatstream.utils.featurefusion.diff_gaussian_rasterization': 'submodules/featurefusion/diff_gaussian_rasterization',
    'instantsplatstream.utils.motionfusion.diff_gaussian_rasterization': 'submodules/motionfusion/diff_gaussian_rasterization',
}

cxx_compiler_flags = []
nvcc_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")
    nvcc_compiler_flags.append("-allow-unsupported-compiler")

setup(
    name="instantsplatstream",
    packages=packages + packages_dot + list(rasterizor_packages.keys()),
    package_dir={
        'instantsplatstream': 'instantsplatstream',
        'dot': 'submodules/dot/dot',
        **rasterizor_packages
    },
    ext_modules=[
        CUDAExtension(
            name="instantsplatstream.utils.featurefusion.diff_gaussian_rasterization._C",
            sources=[os.path.join(featurefusion_root, source) for source in rasterizor_sources],
            extra_compile_args={"nvcc": nvcc_compiler_flags + ["-I" + os.path.join(os.path.abspath(featurefusion_root), "third_party/glm/")]}
        ),
        CUDAExtension(
            name="instantsplatstream.utils.motionfusion.diff_gaussian_rasterization._C",
            sources=[os.path.join(motionfusion_root, source) for source in rasterizor_sources],
            extra_compile_args={"nvcc": nvcc_compiler_flags + ["-I" + os.path.join(os.path.abspath(motionfusion_root), "third_party/glm/")]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
