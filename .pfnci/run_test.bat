@echo off

set CUDA_VER=%1
set PY_VER=%2
set GPU_LIMIT=%3
set CUDA_PATH=CUDA_PATH_V%CUDA_VER%
set PY_PATH=C:\Development\Python\Python%PY_VER%
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PY_PATH%;%PY_PATH%\Scripts\%PATH%

set PYTEST_ATTR=not slow and not ideep

curl -o xpytest.exe --insecure -L https://github.com/disktnk/xpytest/releases/download/v0.0.1/xpytest.exe

if %GPU_LIMIT% gtr 0 (
    rem cannot set variable within if block, so use subroutine
    cmd /c .pfnci\install_cupy.bat
) else (
    set PYTEST_ATTR=not gpu and not cudnn and %PYTEST_ATTR%
)

pip install -e .[test] -vvv
pip install scipy

set CHAINER_TEST_GPU_LIMIT=%GPU_LIMIT%
xpytest --python python -m "%PYTEST_ATTR%" --thread 8 --hint .pfnci/hint_win.pbtxt tests/chainer_tests/**/test_*.py
