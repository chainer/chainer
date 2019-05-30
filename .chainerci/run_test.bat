@echo off

set CUDA_VER=%1
set PY_VER=%2
set CUDA_PATH=CUDA_PATH_V%CUDA_VER%
set PY_PATH=C:\Development\Python\Python%PY_VER%
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PY_PATH%;%PY_PATH%\Scripts\%PATH%

set CHAINER_LATEST_MAJOR_VER=7
set TEST_HOME=%CD%

curl -o xpytest.exe --insecure -L https://github.com/disktnk/xpytest/releases/download/v0.0.1/xpytest.exe

set GET_VER_CMD="import os, imp;print(imp.load_source('_version', os.path.join('chainer', '_version.py')).__version__.split('.')[0])"
for /f "DELIMS=" %%v in (' ^
python -c %GET_VER_CMD% ^
') do set CHAINER_MAJOR_VER=%%v

echo %CHAINER_MAJOR_VER%
cd ..
if %CHAINER_MAJOR_VER% equ %CHAINER_LATEST_MAJOR_VER% (
    git clone https://github.com/cupy/cupy.git
) else (
    git clone -b v%CHAINER_MAJOR_VER% https://github.com/cupy/cupy.git
)
cd cupy
pip install cython
pip install -e . -vvv

cd %TEST_HOME%

pip install -e .[test] -vvv
pip install scipy

set CHAINER_TEST_GPU_LIMIT=1
xpytest --python python -m "not slow and not ideep" --thread 8 --hint .chainerci/hint.pbtxt tests/chainer_tests/**/test_*.py
