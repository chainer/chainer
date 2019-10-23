"""Example for exporting ResNet50 model to ONNX graph.

  $ pwd
  /path/to/onnx-chainer
  $ python examples/resnet50/export.py -I target.jpg -O onnx_model

'model.onnx' will be output under 'onnx_model' directory.
"""
import argparse
import os

import chainer.cuda
import chainercv.links as C
from chainercv.transforms import center_crop
from chainercv.transforms import scale
from chainercv.utils import read_image
from onnx_chainer import export
from onnx_chainer import export_testcase


def export_onnx(input_image_path, output_path, gpu, only_output=True):
    """Export ResNet50 model to ONNX graph

    'model.onnx' file will be exported under ``output_path``.
    """
    model = C.ResNet50(pretrained_model='imagenet', arch='fb')

    input_image = read_image(input_image_path)
    input_image = scale(input_image, 256)
    input_image = center_crop(input_image, (224, 224))
    input_image -= model.mean
    input_image = input_image[None, :]

    if gpu >= 0:
        model.to_gpu()
        input_image = chainer.cuda.to_gpu(input_image)

    if only_output:
        os.makedirs(output_path, exist_ok=True)
        name = os.path.join(output_path, 'model.onnx')
        export(model, input_image, filename=name)
    else:
        # an input and output given by Chainer will be also emitted
        # for using as test dataset
        export_testcase(model, input_image, output_path)


if __name__ == '__main__':
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    default_image_path = os.path.normpath(
        os.path.join(this_file_path, '..', 'images'))
    default_input_path = os.path.join(default_image_path, 'cat.jpg')
    default_output_path = os.path.join('out', 'test_resnet50')

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-G', type=int, default=-1)
    parser.add_argument('--input-image', '-I', default=default_input_path)
    parser.add_argument('--output', '-O', default=default_output_path)
    parser.add_argument('--enable-value-check', '-T', action='store_true')
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    export_onnx(
        args.input_image, args.output, args.gpu, not args.enable_value_check)

    if args.enable_value_check:
        from onnx_chainer.testing.test_onnxruntime import check_model_expect  # NOQA
        check_model_expect(args.output)
