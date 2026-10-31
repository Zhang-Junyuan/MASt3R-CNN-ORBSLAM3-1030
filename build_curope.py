"""Build the current project's RoPE CUDA kernel, without modifying shared packages."""
from pathlib import Path
import os
os.environ.setdefault('CUDA_HOME','/usr/local/cuda')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
os.environ.setdefault('MAX_JOBS','2')
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
root=Path(__file__).resolve().parent
source=root/'third_party/mast3r/dust3r/croco/models/curope'
os.chdir(source)
setup(name='curope',ext_modules=[CUDAExtension(name='curope',sources=['curope.cpp','kernels.cu'],extra_compile_args={'cxx':['-O3'],'nvcc':['-O3','--use_fast_math']})],cmdclass={'build_ext':BuildExtension})
