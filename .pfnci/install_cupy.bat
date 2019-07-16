@echo off

set CHAINER_LATEST_MAJOR_VER=7

rem get chainer version and set to variable using for-do statement
set GET_VER_CMD="import os, imp;print(imp.load_source('_version', os.path.join('chainer', '_version.py')).__version__.split('.')[0])"
for /f "DELIMS=" %%v in (' ^
python -c %GET_VER_CMD% ^
') do set CHAINER_MAJOR_VER=%%v
echo chainer version: %CHAINER_MAJOR_VER%

pushd ..
if %CHAINER_MAJOR_VER% equ %CHAINER_LATEST_MAJOR_VER% (
    git clone https://github.com/cupy/cupy.git
) else (
    git clone -b v%CHAINER_MAJOR_VER% https://github.com/cupy/cupy.git
)
cd cupy
pip install cython
pip install -e . -vvv

popd
