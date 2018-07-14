import hashlib
import os
import shutil
import sys

import filelock
from six.moves.urllib import request

from chainer import utils


_dataset_root = os.environ.get(
    'CHAINER_DATASET_ROOT',
    os.path.join(os.path.expanduser('~'), '.chainer', 'dataset'))


def get_dataset_root():
    """Gets the path to the root directory to download and cache datasets.

    Returns:
        str: The path to the dataset root directory.

    """
    return _dataset_root


def set_dataset_root(path):
    """Sets the root directory to download and cache datasets.

    There are two ways to set the dataset root directory. One is by setting the
    environment variable ``CHAINER_DATASET_ROOT``. The other is by using this
    function. If both are specified, one specified via this function is used.
    The default dataset root is ``$HOME/.chainer/dataset``.

    Args:
        path (str): Path to the new dataset root directory.

    """
    global _dataset_root
    _dataset_root = path


def get_dataset_directory(dataset_name, create_directory=True):
    """Gets the path to the directory of given dataset.

    The generated path is just a concatenation of the global root directory
    (see :func:`set_dataset_root` for how to change it) and the dataset name.
    The dataset name can contain slashes, which are treated as path separators.

    Args:
        dataset_name (str): Name of the dataset.
        create_directory (bool): If True (default), this function also creates
            the directory at the first time. If the directory already exists,
            then this option is ignored.

    Returns:
        str: Path to the dataset directory.

    """
    path = os.path.join(_dataset_root, dataset_name)
    if create_directory:
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


def cached_download(url):
    """Downloads a file and caches it.

    It downloads a file from the URL if there is no corresponding cache. After
    the download, this function stores a cache to the directory under the
    dataset root (see :func:`set_dataset_root`). If there is already a cache
    for the given URL, it just returns the path to the cache without
    downloading the same file.

    .. note::
        This function raises :class:`OSError` when it fails to create
        the cache directory. In older version, it raised :class:`RuntimeError`.

    Args:
        url (str): URL to download from.

    Returns:
        str: Path to the downloaded file.

    """
    cache_root = os.path.join(_dataset_root, '_dl_cache')
    try:
        os.makedirs(cache_root)
    except OSError:
        if not os.path.isdir(cache_root):
            raise

    lock_path = os.path.join(cache_root, '_dl_lock')
    urlhash = hashlib.md5(url.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_root, urlhash)

    with filelock.FileLock(lock_path):
        if os.path.exists(cache_path):
            return cache_path

    with utils.tempdir(dir=cache_root) as temp_root:
        temp_path = os.path.join(temp_root, 'dl')
        sys.stderr.write('Downloading from {}...\n'.format(url))
        sys.stderr.flush()
        request.urlretrieve(url, temp_path)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, cache_path)

    return cache_path


def cache_or_load_file(path, creator, loader):
    """Caches a file if it does not exist, or loads it otherwise.

    This is a utility function used in dataset loading routines. The
    ``creator`` creates the file to given path, and returns the content. If the
    file already exists, the ``loader`` is called instead, and it loads the
    file and returns the content.

    Note that the path passed to the creator is temporary one, and not same as
    the path given to this function. This function safely renames the file
    created by the creator to a given path, even if this function is called
    simultaneously by multiple threads or processes.

    Args:
        path (str): Path to save the cached file.
        creator: Function to create the file and returns the content. It takes
            a path to temporary place as the argument. Before calling the
            creator, there is no file at the temporary path.
        loader: Function to load the cached file and returns the content.

    Returns:
        It returns the returned values by the creator or the loader.

    """
    if os.path.exists(path):
        return loader(path)

    try:
        os.makedirs(_dataset_root)
    except OSError:
        if not os.path.isdir(_dataset_root):
            raise RuntimeError('cannot create dataset directory')

    lock_path = os.path.join(_dataset_root, '_create_lock')

    with utils.tempdir() as temp_dir:
        file_name = os.path.basename(path)
        temp_path = os.path.join(temp_dir, file_name)
        content = creator(temp_path)
        with filelock.FileLock(lock_path):
            if not os.path.exists(path):
                shutil.move(temp_path, path)

    return content
