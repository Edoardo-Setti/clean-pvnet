import os

cuda_home = os.environ.get('CUDA_HOME', '/usr')
cuda_include = os.path.join(cuda_home, 'include')
cuda_lib = os.path.join(cuda_home, 'lib64', 'libcudart.so')
if not os.path.exists(cuda_lib):
    cuda_lib = '/usr/lib/x86_64-linux-gnu/libcudart.so'
ccbin = os.environ.get('CC', 'gcc')
cuda_arch = os.environ.get('PVNET_CUDA_ARCH', 'sm_86')
os.system(
    'nvcc src/nearest_neighborhood.cu -c -o src/nearest_neighborhood.cu.o '
    '-x cu -Xcompiler -fPIC -O2 -arch={} -ccbin {} -I {}'.format(
        cuda_arch, ccbin, cuda_include
    )
)

from cffi import FFI
ffibuilder = FFI()


with open(os.path.join(os.path.dirname(__file__), "src/ext.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source(
    "_ext",
    """
    #include "src/ext.h"
    """,
    extra_objects=['src/nearest_neighborhood.cu.o',
                   cuda_lib],
    libraries=['stdc++']
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    os.system("rm src/*.o")
    os.system("rm *.o")
