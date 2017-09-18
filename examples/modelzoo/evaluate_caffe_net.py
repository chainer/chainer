#!/usr/bin/env python
"""Example code of evaluating a Caffe reference model for ILSVRC2012 task.

Prerequisite: To run this example, crop the center of ILSVRC2012 validation
images and scale them to 256x256, and make a list of space-separated CSV each
column of which contains a full path to an image at the fist column and a zero-
origin label at the second column (this format is same as that used by Caffe's
ImageDataLayer).

"""
from __future__ import print_function
import argparse
import os
import sys
import AlexLike

import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe

# =========================================================
def copy_model(src, dst):
    assert isinstance(src, chainer.Chain)
    assert isinstance(dst, chainer.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, chainer.Chain):
            copy_model(child, dst_child)
            if isinstance(child, chainer.Link):
                match = True
                for a, b in zip(child.namedparams(), dst_child.namedparams()):
                    if a[0] != b[0]:
                        match = False
                        break
                    if a[1].data.shape != b[1].data.shape:
                        match = False
                        break
                    if not match:
                        print('Ignore %s because of parameter mismatch' % child.name)
                        continue
                    for a, b in zip(child.namedparams(), dst_child.namedparams()):
                        b[1].data = a[1].data
                        print('Copy %s' % child.name)
# =========================================================


parser = argparse.ArgumentParser(
    description='Evaluate a Caffe reference model on ILSVRC2012 dataset')
parser.add_argument('dataset', help='Path to validation image-label list file')
parser.add_argument('model_type',
                    choices=('alexnet', 'caffenet', 'googlenet', 'resnet'),
                    help='Model type (alexnet, caffenet, googlenet, resnet)')
parser.add_argument('model', help='Path to the pretrained Caffe model')
parser.add_argument('--basepath', '-b', default='/',
                    help='Base path for images in the dataset')
parser.add_argument('--mean', '-m', default='ilsvrc_2012_mean.npy',
                    help='Path to the mean file')
parser.add_argument('--batchsize', '-B', type=int, default=100,
                    help='Minibatch size')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='Zero-origin GPU ID (nevative value indicates CPU)')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np
assert args.batchsize > 0

chainer.config.train = False  # All the codes will run in test mode


dataset = []
with open(args.dataset) as list_file:
    for line in list_file:
        pair = line.strip().split()
        path = os.path.join(args.basepath, pair[0])
        dataset.append((path, np.int32(pair[1])))

assert len(dataset) % args.batchsize == 0


# =========================================================
model = L.Classifier(alexLike.AlexLike())
original_model = pickle.load(open("alexnet.pkl", "rb"))
copy_model(original_model, model)
# =========================================================


print('Loading Caffe model file %s...' % args.model, file=sys.stderr)
# func = caffe.CaffeFunction(args.model)
func = caffe.CaffeFunction(model)
print('Loaded', file=sys.stderr)
if args.gpu >= 0:
    cuda.get_device_from_id(args.gpu).use()
    func.to_gpu()

if args.model_type == 'alexnet' or args.model_type == 'caffenet':
    in_size = 227
    mean_image = np.load(args.mean)

    def forward(x, t):
        y, = func(inputs={'data': x}, outputs=['fc8'])
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
elif args.model_type == 'googlenet':
    in_size = 224
    # Constant mean over spatial pixels
    mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
    mean_image[0] = 104
    mean_image[1] = 117
    mean_image[2] = 123

    def forward(x, t):
        y, = func(inputs={'data': x}, outputs=['loss3/classifier'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool'])
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
elif args.model_type == 'resnet':
    in_size = 224
    mean_image = np.load(args.mean)

    def forward(x, t):
        y, = func(inputs={'data': x}, outputs=['prob'])
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


cropwidth = 256 - in_size
start = cropwidth // 2
stop = start + in_size
mean_image = mean_image[:, start:stop, start:stop].copy()

x_batch = np.ndarray((args.batchsize, 3, in_size, in_size), dtype=np.float32)
y_batch = np.ndarray((args.batchsize,), dtype=np.int32)

i = 0
count = 0
accum_loss = 0
accum_accuracy = 0
for path, label in dataset:
    print(np.asarray(Image.open(path)).shape)
    image = np.asarray(Image.open(path).convert('RGB')).transpose(2, 0, 1)[::-1]
    image = image[:, start:stop, start:stop].astype(np.float32)
    image -= mean_image

    x_batch[i] = image
    y_batch[i] = label
    i += 1

    if i == args.batchsize:
        x = xp.asarray(x_batch)
        y = xp.asarray(y_batch)

        with chainer.no_backprop_mode():
            loss, accuracy = forward(x, y)

        accum_loss += float(loss.data) * args.batchsize
        accum_accuracy += float(accuracy.data) * args.batchsize
        del x, y, loss, accuracy

        count += args.batchsize
        print('{} / {}'.format(count, len(dataset)), end='\r', file=sys.stderr)
        sys.stderr.flush()

        i = 0


print('mean loss:     {}'.format(accum_loss / count))
print('mean accuracy: {}'.format(accum_accuracy / count))
