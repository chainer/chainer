@echo off

rem ref https://github.com/chainer/chainer-test/blob/2b809ed767b219e8a66734e661d879a97486f014/version.py#L80
rem 1: master branch, 0: backport (stable) branch
git log | findstr /r /c:"^ *\[backport\]" > nul
set IS_MASTER_BRANCH=%ERRORLEVEL%

rem get chainer version and set to variable using for-do statement
set GET_VER_CMD="import os, imp;print(imp.load_source('_version', os.path.join('chainer', '_version.py')).__version__.split('.')[0])"
for /f "DELIMS=" %%v in (' ^
python -c %GET_VER_CMD% ^
') do set CHAINER_MAJOR_VER=%%v
echo chainer version: %CHAINER_MAJOR_VER%

pushd ..
if %IS_MASTER_BRANCH% equ 0 (
    git clone -b v%CHAINER_MAJOR_VER% https://github.com/cupy/cupy.git
) else (
    git clone https://github.com/cupy/cupy.git
)
cd cupy
pip install cython
pip install -e . -vvv

popd
