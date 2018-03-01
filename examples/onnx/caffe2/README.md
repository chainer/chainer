# Copy Chainer models to Caffe2 via ONNX

`export.py` creates a VGG16 model in Chainer, and saves it as ONNX binary, then loads it from Caffe2 using [onnx-caffe2](https://github.com/onnx/onnx-caffe2).
