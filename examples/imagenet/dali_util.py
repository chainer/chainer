import numpy as np

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


import chainer
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
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        if std is None:
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        if num_samples is None:
            initial_fill = 4096
        else:
            initial_fill = min(4096, num_samples)
        self.loader = ops.FileReader(file_root=file_root, file_list=file_list,
                                     random_shuffle=random_shuffle,
                                     initial_fill=initial_fill)
        self.decode = ops.HostDecoder()
        self.resize = ops.Resize(device='gpu', resize_x=256, resize_y=256)
        # self.hue = ops.Hue(device="gpu")
        # self.bright = ops.Brightness(device="gpu")
        # self.cntrst = ops.Contrast(device="gpu")
        # self.rotate = ops.Rotate(device="gpu")
        # self.jitter = ops.Jitter(device="gpu")
        random_area = (crop_size[0] / 256) * (crop_size[1] / 256)
        random_area = _pair(random_area)
        random_aspect_ratio = _pair(1.0)
        self.rrcrop = ops.RandomResizedCrop(
            device='gpu', size=crop_size, random_area=random_area,
            random_aspect_ratio=random_aspect_ratio)
        self.cmnorm = ops.CropMirrorNormalize(
            device='gpu', crop=list(crop_size), mean=mean, std=std)
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
        return images, labels


class DaliPipelineVal(pipeline.Pipeline):

    def __init__(self, file_list, file_root, crop_size,
                 batch_size, num_threads, device_id,
                 random_shuffle=False, seed=-1, mean=None, std=None,
                 num_samples=None):
        super(DaliPipelineVal, self).__init__(batch_size, num_threads,
                                              device_id, seed=seed)
        crop_size = _pair(crop_size)
        if mean is None:
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        if std is None:
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        if num_samples is None:
            initial_fill = 512
        else:
            initial_fill = min(512, num_samples)
        self.loader = ops.FileReader(file_root=file_root, file_list=file_list,
                                     random_shuffle=random_shuffle,
                                     initial_fill=initial_fill)
        self.decode = ops.HostDecoder()
        self.resize = ops.Resize(device='gpu', resize_x=256, resize_y=256)
        self.cmnorm = ops.CropMirrorNormalize(
            device='gpu', crop=list(crop_size), mean=mean, std=std)

    def define_graph(self):
        jpegs, labels = self.loader()
        images = self.decode(jpegs)
        images = self.resize(images.gpu())
        images = self.cmnorm(images)
        return images, labels


class DaliConverter(object):

    def __init__(self, mean, crop_size):
        self.mean = mean
        self.crop_size = crop_size

        ch_mean = np.average(mean, axis=(1, 2))
        perturbation = (mean - ch_mean.reshape(3, 1, 1)) / 255.0
        perturbation = perturbation[:3, :crop_size, :crop_size].astype(
            chainer.get_dtype())
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
                x = x_cupy.astype(chainer.get_dtype())
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
            x = x_cupy.astype(chainer.get_dtype())
            if device is not None and device < 0:
                x = cuda.to_cpu(x)
        else:
            raise ValueError('Unexpected object')
        outputs.append(x)
    return tuple(outputs)
