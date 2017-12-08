from __future__ import print_function

import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt

import argparse
import json
import numpy as np

import chainer
from chainer.dataset.convert import concat_examples
from chainer import cuda
from chainer import serializers

import nets


def save_images(xs, filename):
    width = xs[0].shape[0]
    height = len(xs)

    xs = np.concatenate([cuda.to_cpu(x) for x in xs], axis=0)
    fig, ax = plt.subplots(
        height, width, figsize=(1 * width / 2.5, height / 2.5))
    for i, (ai, xi) in enumerate(zip(ax.ravel(), xs)):
        ai.set_xticklabels([])
        ai.set_yticklabels([])
        ai.set_axis_off()
        color = 'Greens_r' if i // width == 0 else 'Blues_r'  # colored rows
        ai.imshow(xi.reshape(28, 28), cmap=color, vmin=0., vmax=1.)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(filename, bbox_inches='tight', pad=0.)
    plt.clf()
    plt.close('all')


def visualize_reconstruction(model, x, t, filename='vis.png'):
    vs_norm, vs = model.output(x)
    x_recon = model.reconstruct(vs, t)
    save_images([x, x_recon.data], filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CapsNet: MNIST reconstruction')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--load', required=True)
    parser.add_argument('--save-image', default='vis.png')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    model = nets.CapsNet(use_reconstruction=True)
    serializers.load_npz(args.load, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    _, test = chainer.datasets.get_mnist(ndim=3)

    batch = test[:20]
    x, t = concat_examples(batch, args.gpu)

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            visualize_reconstruction(model, x, t, args.save_image)
