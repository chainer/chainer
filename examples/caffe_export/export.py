import argparse
import os

import numpy as np

import chainer
from chainer.exporters import caffe
from chainer.links.model.vision import googlenet
from chainer.links.model.vision import resnet
from chainer.links.model.vision import vgg

archs = {
    'googlenet': googlenet.GoogLeNet,
    'resnet50': resnet.ResNet50Layers,
    'resnet101': resnet.ResNet101Layers,
    'resnet152': resnet.ResNet152Layers,
    'vgg16': vgg.VGG16Layers,
}


class DumpModel(chainer.Chain):

    def __init__(self, arch_name):
        super(DumpModel, self).__init__()
        with self.init_scope():
            self.base_model = archs[arch_name]()

    def forward(self, img):
        return self.base_model(img, layers=['prob'])['prob']


def get_network_for_imagenet(arch_name):
    model = DumpModel(arch_name)
    input_image = np.ones((1, 3, 224, 224), dtype=np.float32)
    input = chainer.Variable(input_image)
    return model, input


def main():
    parser = argparse.ArgumentParser(description='Export')
    parser.add_argument(
        '--arch', '-a', type=str, required=True,
        choices=archs.keys(),
        help='Arch name. models: ' + ', '.join(archs.keys()) + '.')
    parser.add_argument(
        '--out-dir', '-o', type=str, required=True,
        help='Output directory name. '
             'chainer_model.prototxt, chainer_model.caffemodel'
             ' will be created in it')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        print('Created output directory: ' + args.out_dir)
        os.mkdir(args.out_dir)
    else:
        print('Overwriting the existing directory: ' + args.out_dir)
    if not os.path.isdir(args.out_dir):
        raise ValueError(args.out_dir + ' exists but not a directory!')

    print('load model')
    model, input = get_network_for_imagenet(args.arch)

    print('convert to caffe model')
    caffe.export(model, [input], args.out_dir, True)


if __name__ == '__main__':
    main()
