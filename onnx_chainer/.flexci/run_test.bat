@echo off

set CUDA_VER=%1
set PY_VER=%2
set CUDA_PATH=CUDA_PATH_V%CUDA_VER%
set PY_PATH=C:\Development\Python\Python%PY_VER%
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PY_PATH%;%PY_PATH%\Scripts\%PATH%

pip install --pre cupy-cuda100
pip install -e .[test]
pip install packaging "onnx<1.6.0" onnxruntime pytest-cov
pip list -v

pytest -x -s -vvvs tests\onnx_chainer_tests --cov onnx_chainer
