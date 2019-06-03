import os

import numpy
try:
    from scipy import io
    _scipy_available = True
except Exception as e:
    _error = e
    _scipy_available = False

import chainer
from chainer.dataset import download
from chainer.datasets import tuple_dataset


def get_svhn(withlabel=True, scale=1., dtype=None, label_dtype=numpy.int32,
             add_extra=False):
    """Gets the SVHN dataset.

    `The Street View House Numbers (SVHN) dataset
    <http://ufldl.stanford.edu/housenumbers/>`_
    is a dataset similar to MNIST but composed of cropped images of house
    numbers.
    The functionality of this function is identical to the counterpart for the
    MNIST dataset (:func:`~chainer.datasets.get_mnist`),
    with the exception that there is no ``ndim`` argument.

    .. note::
       `SciPy <https://www.scipy.org/>`_ is required to use this feature.

    Args:
        withlabel (bool): If ``True``, it returns datasets with labels. In this
            case, each example is a tuple of an image and a label. Otherwise,
            the datasets only contain images.
        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.
        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is
            used by default (see :ref:`configuration`).
        label_dtype: Data type of the labels.
        add_extra: Use extra training set.

    Returns:
        If ``add_extra`` is ``False``, a tuple of two datasets (train and
        test). Otherwise, a tuple of three datasets (train, test, and extra).
        If ``withlabel`` is ``True``, all datasets are
        :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both
        datasets are arrays of images.

    """
    if not _scipy_available:
        raise RuntimeError('SciPy is not available: %s' % _error)

    train_raw = _retrieve_svhn_training()
    dtype = chainer.get_dtype(dtype)

    train = _preprocess_svhn(train_raw, withlabel, scale, dtype,
                             label_dtype)
    test_raw = _retrieve_svhn_test()
    test = _preprocess_svhn(test_raw, withlabel, scale, dtype,
                            label_dtype)
    if add_extra:
        extra_raw = _retrieve_svhn_extra()
        extra = _preprocess_svhn(extra_raw, withlabel, scale, dtype,
                                 label_dtype)
        return train, test, extra
    else:
        return train, test


def _preprocess_svhn(raw, withlabel, scale, image_dtype, label_dtype):
    images = raw['x'].transpose(3, 2, 0, 1)
    images = images.astype(image_dtype)
    images *= scale / 255.

    labels = raw['y'].astype(label_dtype).flatten()
    # labels go from 1-10, with the digit "0" having label 10.
    # Set "0" to be label 0 to restore expected ordering
    labels[labels == 10] = 0

    if withlabel:
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images


def _retrieve_svhn_training():
    url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    return _retrieve_svhn('train.npz', url)


def _retrieve_svhn_test():
    url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    return _retrieve_svhn('test.npz', url)


def _retrieve_svhn_extra():
    url = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
    return _retrieve_svhn('extra.npz', url)


def _retrieve_svhn(name, url):
    root = download.get_dataset_directory('pfnet/chainer/svhn')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url), numpy.load)


def _make_npz(path, url):
    _path = download.cached_download(url)
    raw = io.loadmat(_path)
    images = raw['X'].astype(numpy.uint8)
    labels = raw['y'].astype(numpy.uint8)

    numpy.savez_compressed(path, x=images, y=labels)
    return {'x': images, 'y': labels}
