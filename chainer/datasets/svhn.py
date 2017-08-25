import os

import numpy
try:
    from scipy import io
    _scipy_available = True
except ImportError:
    _scipy_available = False

from chainer.dataset import download
from chainer.datasets import tuple_dataset


def get_svhn(withlabel=True, scale=1., dtype=numpy.float32,
             label_dtype=numpy.int32):
    """Gets the SVHN dataset.

    `SVHN <http://ufldl.stanford.edu/housenumbers/>` is a dataset
    similar to MNIST but composed of cropped images of house numbers.
    The functionality is identical to the MNIST dataset,
    with the exception that there is no ``ndim`` argument.

    Args:
        withlabel (bool): If ``True``, it returns datasets with labels. In this
            case, each example is a tuple of an image and a label. Otherwise,
            the datasets only contain images.
        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.
        dtype: Data type of resulting image arrays.
        label_dtype: Data type of the labels.

    Returns:
        A tuple of two datasets. If ``withlabel`` is ``True``, both datasets
        are :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both
        datasets are arrays of images.

    """
    if not _scipy_available:
        raise RuntimeError('scipy is not available')

    train_raw = _retrieve_svhn_training()
    train = _preprocess_svhn(train_raw, withlabel, scale, dtype,
                             label_dtype)
    test_raw = _retrieve_svhn_test()
    test = _preprocess_svhn(test_raw, withlabel, scale, dtype,
                            label_dtype)
    return train, test


def _preprocess_svhn(raw, withlabel, scale, image_dtype, label_dtype):
    images = raw["x"].transpose(3, 2, 0, 1)
    images = images.astype(image_dtype)
    images *= scale / 255.

    labels = raw["y"].astype(label_dtype).flatten()
    # labels go from 1-10, with the digit "0" having label 10.
    # Set "0" to be label 0 to restore expected ordering
    labels[labels == 10] = 0

    if withlabel:
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images


def _retrieve_svhn_training():
    url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    return _retrieve_svhn("train.npz", url)


def _retrieve_svhn_test():
    url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    return _retrieve_svhn("test.npz", url)


def _retrieve_svhn(name, url):
    root = download.get_dataset_directory('pfnet/chainer/svhn')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url), numpy.load)


def _make_npz(path, url):
    _path = download.cached_download(url)
    raw = io.loadmat(_path)
    images = raw["X"].astype(numpy.uint8)
    labels = raw["y"].astype(numpy.uint8)

    numpy.savez_compressed(path, x=images, y=labels)
    return {'x': images, 'y': labels}
