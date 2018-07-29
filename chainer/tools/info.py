import chainer
import os
import platform
import subprocess
import re
try:
    import cupy
    cupy_version = cupy.__version__
except ImportError:
    cupy_version = 'not installed'

try:
    with open(os.devnull, 'w') as stderr:
        nvcc_output = subprocess.check_output("nvcc --version",
                                              stderr=stderr,
                                              shell=True)
    nvcc_output = nvcc_output.decode('utf-8')

    if nvcc_output:
        nvcc_output = nvcc_output.rstrip('\n')
        *_, nvcc_version_line = nvcc_output.split('\n')
        nvcc_version = re.search(r'release (.*)', nvcc_version_line)
        nvcc_version = nvcc_version.group(1)

except Exception:
    nvcc_version = 'not installed'

print('- Chainer version: ', chainer.__version__)
print('- CuPy version: ', cupy_version)
print('- OS/Platform: ', platform.platform())
print('- CUDA version: ', nvcc_version)
