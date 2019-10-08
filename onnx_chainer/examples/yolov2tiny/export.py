"""Example for exporting YOLOv2 Tiny model to ONNX graph.

  $ pwd
  /path/to/onnx-chainer
  $ python examples/yolov2tiny/export.py -I target.jpg -O onnx_model

'model.onnx' will be output under 'onnx_model' directory.

NOTE: Outputs are required postprocessing to draw bbox on the target.jpg.
      See ChainerCV's example of detection 'visualize_models.py'.
"""
import argparse
import os

import chainer.cuda
from chainercv.experimental.links import YOLOv2Tiny
from chainercv.utils import read_image
from onnx_chainer import export
from onnx_chainer import export_testcase


def export_onnx(input_image_path, output_path, gpu, only_output=True):
    """Export YOLOv2 Tiny model to ONNX graph

    'model.onnx' file will be exported under ``output_path``.
    """
    model = YOLOv2Tiny(pretrained_model='voc0712')

    input_image = read_image(input_image_path)
    input_image = input_image[None, :]

    if gpu >= 0:
        model.to_gpu()
        input_image = chainer.cuda.to_gpu(input_image)

    if only_output:
        os.makedirs(output_path, exist_ok=True)
        name = os.path.join(output_path, 'model.onnx')
        export(
            model, input_image, filename=name,
            output_names=('locs', 'objs', 'confs'))
    else:
        # an input and output given by Chainer will be also emitted
        # for using as test dataset
        export_testcase(
            model, input_image, output_path,
            output_names=('locs', 'objs', 'confs'))


if __name__ == '__main__':
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    default_image_path = os.path.normpath(
        os.path.join(this_file_path, '..', 'images'))
    default_input_path = os.path.join(default_image_path, 'cat.jpg')
    default_output_path = os.path.join('out', 'test_yolo2tiny')

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
        check_model_expect(args.output, atol=1e-3)
