# Caffe Export

This example shows how to export Chainer models into Caffe protobuf format. With this example, you can easily export the following five pre-trained models:

- GoogLeNet
- ResNet50
- ResNet101
- ResNet152
- VGG16

The implementation of these models are found in [chainer/links/model/vision](https://github.com/chainer/chainer/tree/v3/chainer/links/model/vision). As you can see in the implementation, these models return dictionaries and the outputs of last layers are stored in `prob` keys. So, in this example, the `DumpModel` class wraps the given model to extract the output in `__call__` method. Please note that this procedure may not be necessary for your own model if it returns the output `Variable` directly from `__call__` method.

# Usage

Run the script `export.py`. You can choose one out of the above five architectures with `--arch` argument. The output directory can be specified with `--out-dir`.

Example:
```
python export.py --arch resnet50 --out-dir ./
```

# Export your own model

To use `chainer.exporters.caffe.export()` method for your own model, please see the details here: [chainer.exporters.caffe.export](https://docs.chainer.org/en/latest/reference/generated/chainer.exporters.caffe.export.html#chainer-exporters-caffe-export).