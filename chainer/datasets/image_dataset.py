import os

import numpy
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e
import bisect
import io
import six
import threading
import zipfile

import chainer
from chainer.dataset import dataset_mixin


def _read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = numpy.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image


def _postprocess_image(image):
    if image.ndim == 2:
        # image is greyscale
        image = image[..., None]
    return image.transpose(2, 0, 1)


class ImageDataset(dataset_mixin.DatasetMixin):

    """Dataset of images built from a list of paths to image files.

    This dataset reads an external image file on every call of the
    :meth:`__getitem__` operator. The paths to the image to retrieve is given
    as either a list of strings or a text file that contains paths in distinct
    lines.

    Each image is automatically converted to arrays of shape
    ``channels, height, width``, where ``channels`` represents the number of
    channels in each pixel (e.g., 1 for grey-scale images, and 3 for RGB-color
    images).

    .. note::
       **This dataset requires the Pillow package being installed.** In order
       to use this dataset, install Pillow (e.g. by using the command ``pip
       install Pillow``). Be careful to prepare appropriate libraries for image
       formats you want to use (e.g. libpng for PNG images, and libjpeg for JPG
       images).

    .. warning::
       **You are responsible for preprocessing the images before feeding them
       to a model.** For example, if your dataset contains both RGB and
       grayscale images, make sure that you convert them to the same format.
       Otherwise you will get errors because the input dimensions are different
       for RGB and grayscale images.

    Args:
        paths (str or list of strs): If it is a string, it is a path to a text
            file that contains paths to images in distinct lines. If it is a
            list of paths, the ``i``-th element represents the path to the
            ``i``-th image. In both cases, each path is a relative one from the
            root path given by another argument.
        root (str): Root directory to retrieve images from.
        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is
            used by default (see :ref:`configuration`).

    """

    def __init__(self, paths, root='.', dtype=None):
        _check_pillow_availability()
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root = root
        self._dtype = chainer.get_dtype(dtype)

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        image = _read_image_as_array(path, self._dtype)

        return _postprocess_image(image)


class LabeledImageDataset(dataset_mixin.DatasetMixin):

    """Dataset of image and label pairs built from a list of paths and labels.

    This dataset reads an external image file like :class:`ImageDataset`. The
    difference from :class:`ImageDataset` is that this dataset also returns a
    label integer. The paths and labels are given as either a list of pairs or
    a text file contains paths/labels pairs in distinct lines. In the latter
    case, each path and corresponding label are separated by white spaces. This
    format is same as one used in Caffe.

    .. note::
       **This dataset requires the Pillow package being installed.** In order
       to use this dataset, install Pillow (e.g. by using the command ``pip
       install Pillow``). Be careful to prepare appropriate libraries for image
       formats you want to use (e.g. libpng for PNG images, and libjpeg for JPG
       images).

    .. warning::
       **You are responsible for preprocessing the images before feeding them
       to a model.** For example, if your dataset contains both RGB and
       grayscale images, make sure that you convert them to the same format.
       Otherwise you will get errors because the input dimensions are different
       for RGB and grayscale images.

    Args:
        pairs (str or list of tuples): If it is a string, it is a path to a
            text file that contains paths to images in distinct lines. If it is
            a list of pairs, the ``i``-th element represents a pair of the path
            to the ``i``-th image and the corresponding label. In both cases,
            each path is a relative one from the root path given by another
            argument.
        root (str): Root directory to retrieve images from.
        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is
            used by default (see :ref:`configuration`).
        label_dtype: Data type of the labels.

    """

    def __init__(self, pairs, root='.', dtype=None, label_dtype=numpy.int32):
        _check_pillow_availability()
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0], int(pair[1])))
        self._pairs = pairs
        self._root = root
        self._dtype = chainer.get_dtype(dtype)
        self._label_dtype = label_dtype

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        image = _read_image_as_array(full_path, self._dtype)

        label = numpy.array(int_label, dtype=self._label_dtype)
        return _postprocess_image(image), label


class LabeledZippedImageDataset(dataset_mixin.DatasetMixin):

    """Dataset of zipped image and label pairs.

    This dataset is zip version of :class:`LabeledImageDataset`. It
    takes a zipfile like :class:`ZippedImageDataset`. The label file
    shall contain lines like text file used in
    :class:`LabeledImageDataset`, but a filename in each line of the
    label file shall match with a file in the zip archive.

    Args:
        zipfilename (str): Path to a zipfile with images
        labelfilename (str): Path to a label file. ``i``-th line shall
            contain a filename and an integer label that corresponds
            to the ``i``-th sample. A filename in the label file shall
            match with a filename in the zip file given with
            `zipfilename`.
        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is
            used by default (see :ref:`configuration`).
        label_dtype: Data type of the labels.

    """

    def __init__(self, zipfilename, labelfilename, dtype=None,
                 label_dtype=numpy.int32):
        _check_pillow_availability()
        pairs = []
        with open(labelfilename) as pairs_file:
            for i, line in enumerate(pairs_file):
                pair = line.strip().split()
                if len(pair) != 2:
                    raise ValueError(
                        'invalid format at line {} in file {}'.format(
                            i, pairs_file))
                pairs.append((pair[0], int(pair[1])))
        self._pairs = pairs
        self._label_dtype = label_dtype
        self._zipfile = ZippedImageDataset(zipfilename, dtype=dtype)

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, int_label = self._pairs[i]
        label = numpy.array(int_label, dtype=self._label_dtype)
        return self._zipfile.get_example(path), label


class MultiZippedImageDataset(dataset_mixin.DatasetMixin):
    """Dataset of images built from a list of paths to zip files.

    This dataset reads an external image file in given zipfiles. The
    zipfiles shall contain only image files.
    This shall be able to replace ImageDataset and works better on NFS
    and other networked file systems. The user shall find good balance
    between zipfile size and number of zipfiles (e.g. granularity)

    Args:
        zipfilenames (list of strings): List of zipped archive filename.
        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is
            used by default (see :ref:`configuration`).
    """

    def __init__(self, zipfilenames, dtype=None):
        self._zfs = [ZippedImageDataset(fn, dtype) for fn in zipfilenames]
        self._zpaths_accumlens = [0]
        zplen = 0
        for zf in self._zfs:
            zplen += len(zf)
            self._zpaths_accumlens.append(zplen)

    def __len__(self):
        return self._zpaths_accumlens[-1]

    def get_example(self, i):
        tgt = bisect.bisect(self._zpaths_accumlens, i) - 1

        lidx = i - self._zpaths_accumlens[tgt]
        return self._zfs[tgt].get_example(lidx)


class ZippedImageDataset(dataset_mixin.DatasetMixin):
    """Dataset of images built from a zip file.

    This dataset reads an external image file in the given
    zipfile. The zipfile shall contain only image files.
    This shall be able to replace ImageDataset and works better on NFS
    and other networked file systems. If zipfile becomes too large you
    may consider ``MultiZippedImageDataset`` as a handy alternative.

    Known issue: pickle and unpickle on same process may cause race
    condition on ZipFile. Pickle of this class is expected to be sent
    to different processess via ChainerMN.

    Args:
        zipfilename (str): a string to point zipfile path
        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is
            used by default (see :ref:`configuration`).

    """

    def __init__(self, zipfilename, dtype=None):
        self._zipfilename = zipfilename
        self._zf = zipfile.ZipFile(zipfilename)
        self._zf_pid = os.getpid()
        self._dtype = chainer.get_dtype(dtype)
        self._paths = [x for x in self._zf.namelist() if not x.endswith('/')]
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._paths)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_zf'] = None
        d['_lock'] = None
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._lock = threading.Lock()

    def get_example(self, i_or_filename):
        # LabeledZippedImageDataset needs file with filename in zip archive
        if isinstance(i_or_filename, six.integer_types):
            zfn = self._paths[i_or_filename]
        else:
            zfn = i_or_filename

        # PIL may seek() on the file -- zipfile won't support it
        with self._lock:
            if self._zf is None or self._zf_pid != os.getpid():
                self._zf_pid = os.getpid()
                self._zf = zipfile.ZipFile(self._zipfilename)
            image_file_mem = self._zf.read(zfn)
        image_file = io.BytesIO(image_file_mem)
        image = _read_image_as_array(image_file, self._dtype)
        return _postprocess_image(image)


def _check_pillow_availability():
    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))
