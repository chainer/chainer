#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

"""
import argparse
import random

import numpy as np

import chainer
from chainer import dataset
from chainer import training
from chainer.training import extensions

import alex
import googlenet
import googlenetbn
import nin
import resnet50
import resnext50

try:
    from nvidia import dali
    from nvidia.dali import ops
    from nvidia.dali import pipeline
    _dali_available = True
except ImportError:
    class pipeline(object):
        Pipeline = object
        pass
    _dali_available = False

from chainer.backends import cuda
import ctypes


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class DaliPipelineTrain(pipeline.Pipeline):

    def __init__(self, file_list, file_root, crop_size,
                 batch_size, num_threads, device_id,
                 random_shuffle=True, seed=-1, mean=None, std=None,
                 num_samples=None):
        super(DaliPipelineTrain, self).__init__(batch_size, num_threads,
                                                device_id, seed=seed)
        crop_size = _pair(crop_size)
        if mean is None:
            mean = (0.485 * 255, 0.456 * 255, 0.406 * 255)
        if std is None:
            std = (0.229 * 255, 0.224 * 255, 0.225 * 255)
        if num_samples is None:
            initial_fill = 4096
        else:
            initial_fill = min(4096, num_samples)
        self.loader = ops.FileReader(file_root=file_root, file_list=file_list,
                                     random_shuffle=random_shuffle,
                                     initial_fill=initial_fill)
        self.decode = ops.HostDecoder()
        self.resize = ops.Resize(device="gpu", resize_a=256, resize_b=256,
                                 warp_resize=True)
        # self.hue = ops.Hue(device="gpu")
        # self.bright = ops.Brightness(device="gpu")
        # self.cntrst = ops.Contrast(device="gpu")
        # self.rotate = ops.Rotate(device="gpu")
        # self.jitter = ops.Jitter(device="gpu")
        random_area = (crop_size[0] / 256) * (crop_size[1] / 256)
        random_area = _pair(random_area)
        random_aspect_ratio = _pair(1.0)
        self.rrcrop = ops.RandomResizedCrop(
            device="gpu", size=crop_size, random_area=random_area,
            random_aspect_ratio=random_aspect_ratio)
        self.cmnorm = ops.CropMirrorNormalize(
            device="gpu", crop=crop_size, mean=mean, std=std)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        jpegs, labels = self.loader()
        images = self.decode(jpegs)
        images = self.resize(images.gpu())
        # images = self.hue(images, hue=ops.Uniform(range=(-3.0, 3.0))())
        # images = self.bright(images,
        #                      brightness=ops.Uniform(range=(0.9, 1.1))())
        # images = self.cntrst(images,
        #                      contrast=ops.Uniform(range=(0.9, 1.1))())
        # images = self.rotate(images,
        #                      angle=ops.Uniform(range=(-5.0, 5.0))())
        # images = self.jitter(images)
        images = self.rrcrop(images)
        images = self.cmnorm(images, mirror=self.coin())
        return [images, labels]


class DaliPipelineVal(pipeline.Pipeline):

    def __init__(self, file_list, file_root, crop_size,
                 batch_size, num_threads, device_id,
                 random_shuffle=False, seed=-1, mean=None, std=None,
                 num_samples=None):
        super(DaliPipelineVal, self).__init__(batch_size, num_threads,
                                              device_id, seed=seed)
        crop_size = _pair(crop_size)
        if mean is None:
            mean = (0.485 * 255, 0.456 * 255, 0.406 * 255)
        if std is None:
            std = (0.229 * 255, 0.224 * 255, 0.225 * 255)
        if num_samples is None:
            initial_fill = 512
        else:
            initial_fill = min(512, num_samples)
        self.loader = ops.FileReader(file_root=file_root, file_list=file_list,
                                     random_shuffle=random_shuffle,
                                     initial_fill=initial_fill)
        self.decode = ops.HostDecoder()
        self.resize = ops.Resize(device="gpu", resize_a=256, resize_b=256,
                                 warp_resize=True)
        self.cmnorm = ops.CropMirrorNormalize(
            device="gpu", crop=crop_size, mean=mean, std=std)

    def define_graph(self):
        jpegs, labels = self.loader()
        images = self.decode(jpegs)
        images = self.resize(images.gpu())
        images = self.cmnorm(images)
        return [images, labels]


class DaliConverter(object):

    def __init__(self, mean, crop_size):
        self.mean = mean
        self.crop_size = crop_size

        ch_mean = np.average(mean, axis=(1, 2))
        perturbation = (mean - ch_mean.reshape(3, 1, 1)) / 255.0
        perturbation = perturbation[:3, :crop_size, :crop_size].astype(
            np.float32)
        self.perturbation = perturbation.reshape(1, 3, crop_size, crop_size)

    def __call__(self, inputs, device=None):
        """Convert DALI arrays to Numpy/CuPy arrays"""

        xp = cuda.get_array_module(self.perturbation)
        if xp is not cuda.cupy:
            self.perturbation = cuda.to_gpu(self.perturbation, device)

        outputs = []
        for i in range(len(inputs)):
            x = inputs[i].as_tensor()
            if (isinstance(x, dali.backend_impl.TensorCPU)):
                x = np.array(x)
                if x.ndim == 2 and x.shape[1] == 1:
                    x = x.squeeze(axis=1)
                if device is not None and device >= 0:
                    x = cuda.to_gpu(x, device)
            elif (isinstance(x, dali.backend_impl.TensorGPU)):
                x_cupy = cuda.cupy.empty(shape=x.shape(), dtype=x.dtype())
                # Synchronization is necessary here to avoid data corruption
                # because DALI and CuPy will use different CUDA streams.
                cuda.cupy.cuda.runtime.deviceSynchronize()
                # copy data from DALI array to CuPy array
                x.copy_to_external(ctypes.c_void_p(x_cupy.data.ptr))
                cuda.cupy.cuda.runtime.deviceSynchronize()
                x = x_cupy
                if self.perturbation is not None:
                    x = x - self.perturbation
                if device is not None and device < 0:
                    x = cuda.to_cpu(x)
            else:
                raise ValueError('Unexpected object')
            outputs.append(x)
        return tuple(outputs)


def dali_converter(inputs, device=None):
    """Convert DALI arrays to Numpy/CuPy arrays"""

    outputs = []
    for i in range(len(inputs)):
        x = inputs[i].as_tensor()
        if (isinstance(x, dali.backend_impl.TensorCPU)):
            x = np.array(x)
            if x.ndim == 2 and x.shape[1] == 1:
                x = x.squeeze(axis=1)
            if device is not None and device >= 0:
                x = cuda.to_gpu(x, device)
        elif (isinstance(x, dali.backend_impl.TensorGPU)):
            x_cupy = cuda.cupy.empty(shape=x.shape(), dtype=x.dtype())
            # Synchronization is necessary here to avoid data corruption
            # because DALI and CuPy will use different CUDA streams.
            cuda.cupy.cuda.runtime.deviceSynchronize()
            # copy data from DALI array to CuPy array
            x.copy_to_external(ctypes.c_void_p(x_cupy.data.ptr))
            cuda.cupy.cuda.runtime.deviceSynchronize()
            x = x_cupy
            if device is not None and device < 0:
                x = cuda.to_cpu(x)
        else:
            raise ValueError('Unexpected object')
        outputs.append(x)
    return tuple(outputs)


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype(np.float32)
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


def main():
    archs = {
        'alex': alex.Alex,
        'alex_fp16': alex.AlexFp16,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
        'resnext50': resnext50.ResNeXt50,
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--dali', action='store_true')
    parser.set_defaults(dali=False)
    args = parser.parse_args()

    # Initialize the model to train
    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from {}'.format(args.initmodel))
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(
            args.gpu).use()  # Make the GPU current
        model.to_gpu()

    # Load the mean file
    mean = np.load(args.mean)
    if args.dali:
        if not _dali_available:
            raise RuntimeError('DALI seems not available on your system.')
        num_threads = args.loaderjob
        if num_threads is None or num_threads <= 0:
            num_threads = 1
        ch_mean = np.average(mean, axis=(1, 2))
        ch_std = (255.0, 255.0, 255.0)
        # Setup DALI pipelines
        train_pipe = DaliPipelineTrain(
            args.train, args.root, model.insize, args.batchsize,
            num_threads, args.gpu, random_shuffle=True,
            mean=ch_mean, std=ch_std)
        val_pipe = DaliPipelineVal(
            args.val, args.root, model.insize, args.val_batchsize,
            num_threads, args.gpu, random_shuffle=False,
            mean=ch_mean, std=ch_std)
        train_iter = chainer.iterators.DaliIterator(train_pipe)
        val_iter = chainer.iterators.DaliIterator(val_pipe, repeat=False)
        # converter = dali_converter
        converter = DaliConverter(mean=mean, crop_size=model.insize)
    else:
        # Load the dataset files
        train = PreprocessedDataset(args.train, args.root, mean, model.insize)
        val = PreprocessedDataset(args.val, args.root, mean, model.insize,
                                  False)
        # These iterators load the images with subprocesses running in parallel
        # to the training/validation.
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=args.loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(
            val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)
        converter = dataset.concat_examples

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (1 if args.test else 100000), 'iteration'
    log_interval = (1 if args.test else 1000), 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, converter=converter,
                                        device=args.gpu), trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
