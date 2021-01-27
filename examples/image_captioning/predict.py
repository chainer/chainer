#!/usr/bin/env python
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

import chainer
from chainer import serializers
import chainerx

import datasets
from model import ImageCaptionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str,
                        help='Image path')
    parser.add_argument('--img-dir', type=str,
                        help='Image directory path, instead of a single image')
    parser.add_argument('--model', type=str, default='result/model_1000',
                        help='Trained model path')
    parser.add_argument('--mscoco-root', type=str, default='data',
                        help='MSOCO dataset root directory')
    parser.add_argument('--rnn', type=str, default='nsteplstm',
                        choices=['nsteplstm', 'lstm'],
                        help='Language model layer type')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--max-caption-length', type=int, default=30,
                        help='Maximum caption length generated')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)
    if device.xp is chainerx:
        sys.stderr.write('This example does not support ChainerX devices.\n')
        sys.exit(1)

    print('Device: {}'.format(device))
    print()

    # Load the dataset to obtain the vocabulary, which is needed to convert
    # predicted tokens into actual words
    train, _ = datasets.get_mscoco(args.mscoco_root)
    vocab = train.vocab
    ivocab = {v: k for k, v in vocab.items()}

    model = ImageCaptionModel(len(train.vocab), rnn=args.rnn)
    serializers.load_npz(args.model, model)

    model.to_device(device)

    if args.img_dir:  # Read all images in directory
        img_paths = [
            i for i in glob.glob(os.path.join(args.img_dir, '*')) if
            i.endswith(('png', 'jpg'))]
        img_paths = sorted(img_paths)
    else:  # Load a single image
        img_paths = [args.img]

    if not img_paths:
        raise IOError('No images found for the given path')

    imgs = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img = model.prepare(img)
        imgs.append(img)
    imgs = np.asarray(imgs)
    imgs = device.send(imgs)

    bos = vocab['<bos>']
    eos = vocab['<eos>']
    with chainer.using_config('train', False), \
            chainer.no_backprop_mode():
        captions = model.predict(
            imgs, bos=bos, eos=eos, max_caption_length=args.max_caption_length)
    captions = chainer.get_device('@numpy').send(captions)

    # Print the predicted captions
    file_names = [os.path.basename(path) for path in img_paths]
    max_length = max(len(name) for name in file_names)
    for file_name, caption in zip(file_names, captions):
        caption = ' '.join(ivocab[token] for token in caption)
        caption = caption.replace('<bos>', '').replace('<eos>', '').strip()
        print(('{0:' + str(max_length) + '} {1}').format(file_name, caption))


if __name__ == '__main__':
    main()
