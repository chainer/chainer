import numpy

import six

from chainer.backends import cuda
from chainer import serializer


class DictionarySerializer(serializer.Serializer):

    """Serializer for dictionary.

    This is the standard serializer in Chainer. The hierarchy of objects are
    simply mapped to a flat dictionary with keys representing the paths to
    objects in the hierarchy.

    .. note::
       Despite of its name, this serializer DOES NOT serialize the
       object into external files. It just build a flat dictionary of arrays
       that can be fed into :func:`numpy.savez` and
       :func:`numpy.savez_compressed`. If you want to use this serializer
       directly, you have to manually send a resulting dictionary to one of
       these functions.

    Args:
        target (dict): The dictionary that this serializer saves the objects
            to. If target is None, then a new dictionary is created.
        path (str): The base path in the hierarchy that this serializer
            indicates.

    Attributes:
        ~DictionarySerializer.target (dict): The target dictionary.
            Once the serialization completes, this dictionary can be fed into
            :func:`numpy.savez` or :func:`numpy.savez_compressed` to serialize
            it in the NPZ format.

    """

    def __init__(self, target=None, path=''):
        self.target = {} if target is None else target
        self.path = path

    def __getitem__(self, key):
        key = key.strip('/')
        return DictionarySerializer(self.target, self.path + key + '/')

    def __call__(self, key, value):
        key = key.lstrip('/')
        ret = value
        if isinstance(value, cuda.ndarray):
            value = value.get()
        arr = numpy.asarray(value)
        self.target[self.path + key] = arr
        return ret


def save_npz(file, obj, compression=True):
    """Saves an object to the file in NPZ format.

    This is a short-cut function to save only one object into an NPZ file.

    Args:
        file (str or file-like): Target file to write to.
        obj: Object to be serialized. It must support serialization protocol.
        compression (bool): If ``True``, compression in the resulting zip file
            is enabled.

    .. seealso::
        :func:`chainer.serializers.load_npz`

    """
    if isinstance(file, six.string_types):
        with open(file, 'wb') as f:
            save_npz(f, obj, compression)
        return

    s = DictionarySerializer()
    s.save(obj)
    if compression:
        numpy.savez_compressed(file, **s.target)
    else:
        numpy.savez(file, **s.target)


class NpzDeserializer(serializer.Deserializer):

    """Deserializer for NPZ format.

    This is the standard deserializer in Chainer. This deserializer can be used
    to read an object serialized by :func:`save_npz`.

    Args:
        npz: `npz` file object.
        path: The base path that the deserialization starts from.
        strict (bool): If ``True``, the deserializer raises an error when an
            expected value is not found in the given NPZ file. Otherwise,
            it ignores the value and skip deserialization.

    """

    def __init__(self, npz, path='', strict=True):
        self.npz = npz
        self.path = path
        self.strict = strict

    def __getitem__(self, key):
        key = key.strip('/')
        return NpzDeserializer(
            self.npz, self.path + key + '/', strict=self.strict)

    def __call__(self, key, value):
        key = self.path + key.lstrip('/')
        if not self.strict and key not in self.npz:
            return value

        dataset = self.npz[key]
        if dataset[()] is None:
            return None

        if value is None:
            return dataset
        elif isinstance(value, numpy.ndarray):
            numpy.copyto(value, dataset)
        elif isinstance(value, cuda.ndarray):
            value.set(numpy.asarray(dataset))
        else:
            value = type(value)(numpy.asarray(dataset))
        return value


def load_npz(file, obj, path='', strict=True):
    """Loads an object from the file in NPZ format.

    This is a short-cut function to load from an `.npz` file that contains only
    one object.

    Args:
        file (str or file-like): File to be loaded.
        obj: Object to be deserialized. It must support serialization protocol.
        path (str): The path in the hierarchy of the serialized data under
            which the data is to be loaded. The default behavior (blank) will
            load all data under the root path.
        strict (bool): If ``True``, the deserializer raises an error when an
            expected value is not found in the given NPZ file. Otherwise,
            it ignores the value and skip deserialization.

    .. seealso::
        :func:`chainer.serializers.save_npz`

    """
    with numpy.load(file) as f:
        d = NpzDeserializer(f, path=path, strict=strict)
        d.load(obj)
