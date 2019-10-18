import numpy
import six

from chainer.backends import _chainerx
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import serializer
import chainerx


# For historical reasons, NPZ serializers in Chainer allow pickle despite their
# potential security issues. This behavior may be changed in future.

# `numpy.save` and `numpy.load` have `allow_pickle` option. `numpy.savez` and
# `numpy.savez_compressed` do not have an option to disable pickle.
# Before NumPy 1.10, pickle was always allowed. Since NumPy 1.16.3, pickle is
# not allowed by default.
_allow_pickle_kwargs = {}
if numpy.lib.NumpyVersion(numpy.__version__) >= '1.10.0':
    _allow_pickle_kwargs['allow_pickle'] = True


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
        self.target[self.path + key] = (
            _cpu._to_cpu(value) if value is not None
            else numpy.asarray(None))
        return value


def serialize(obj):
    """Serializes an object to a dictionary object.

    Args:
        obj: Object to be serialized. It must support serialization protocol.

    Returns:
        dict: Serialized object.
    """
    s = DictionarySerializer()
    s.save(obj)
    return s.target


def save_npz(file, obj, compression=True):
    """Saves an object to the file in NPZ format.

    This is a short-cut function to save only one object into an NPZ file.

    Args:
        file (str or file-like): Target file to write to.
        obj: Object to be serialized. It must support serialization protocol.
            If it is a dictionary object, the serialization will be skipped.
        compression (bool): If ``True``, compression in the resulting zip file
            is enabled.

    .. seealso::
        :func:`chainer.serializers.load_npz`

    """
    if isinstance(file, six.string_types):
        with open(file, 'wb') as f:
            save_npz(f, obj, compression)
        return

    if isinstance(obj, dict):
        target = obj
    else:
        s = DictionarySerializer()
        s.save(obj)
        target = s.target

    if compression:
        numpy.savez_compressed(file, **target)
    else:
        numpy.savez(file, **target)


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
        ignore_names (string, callable or list of them):
            If callable, it is a function that takes a name of a parameter
            and a persistent and returns ``True`` when it needs to be skipped.
            If string, this is a name of a parameter or persistent that are
            going to be skipped.
            This can also be a list of callables and strings that behave as
            described above.

    """

    def __init__(self, npz, path='', strict=True, ignore_names=None):
        self.npz = npz
        self.path = path
        self.strict = strict
        if ignore_names is None:
            ignore_names = []
        self.ignore_names = ignore_names

    def __getitem__(self, key):
        key = key.strip('/')
        return NpzDeserializer(
            self.npz, self.path + key + '/', strict=self.strict,
            ignore_names=self.ignore_names)

    def __call__(self, key, value):
        key = self.path + key.lstrip('/')
        if not self.strict and key not in self.npz:
            return value

        if isinstance(self.ignore_names, (tuple, list)):
            ignore_names = self.ignore_names
        else:
            ignore_names = (self.ignore_names,)
        for ignore_name in ignore_names:
            if isinstance(ignore_name, str):
                if key == ignore_name:
                    return value
            elif callable(ignore_name):
                if ignore_name(key):
                    return value
            else:
                raise ValueError(
                    'ignore_names needs to be a callable, string or '
                    'list of them.')

        dataset = self.npz[key]
        if dataset[()] is None:
            return None
        if value is None:
            return dataset
        if isinstance(value, chainerx.ndarray):
            value[...] = _chainerx._array_to_chainerx(
                numpy.asarray(dataset), value.device)
        elif isinstance(value, numpy.ndarray):
            numpy.copyto(value, dataset)
        elif isinstance(value, cuda.ndarray):
            value.set(numpy.asarray(dataset, dtype=value.dtype))
        elif isinstance(value, intel64.mdarray):
            intel64.ideep.basic_copyto(value, numpy.asarray(dataset))
        else:
            value_type = type(value)
            dataset_arr = numpy.asarray(dataset)
            if (issubclass(dataset_arr.dtype.type, numpy.number)
                    and not (issubclass(dataset_arr.dtype.type, numpy.integer)
                             and value_type in six.integer_types)
                    # Casting a `numpy.integer` scalar by `int()` case above is
                    # safe as `int()` gives unlimited precision integer (it's
                    # also true for `long()`/`int()` on Python 2). For such a
                    # case, the check below may be too strict. For example,
                    # `numpy.can_cast(numpy.int64, int)`, which checks cast-
                    # ability to `dtype(int)`, gives `False` on a platform
                    # whose `dtype(int)` is `numpy.int32` like Windows/x64.
                    and not numpy.can_cast(
                        dataset_arr.dtype, value_type, casting='safe')):
                raise TypeError(
                    'Cannot safely deserialize from numpy array with dtype={} '
                    'into a variable of type {}.'.format(
                        dataset.dtype, type(value)))
            value = value_type(dataset_arr)
        return value


def load_npz(file, obj, path='', strict=True, ignore_names=None):
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
        ignore_names (string, callable or list of them):
            If callable, it is a function that takes a name of a parameter
            and a persistent and returns ``True`` when it needs to be skipped.
            If string, this is a name of a parameter or persistent that are
            going to be skipped.
            This can also be a list of callables and strings that behave as
            described above.

    .. seealso::
        :func:`chainer.serializers.save_npz`

    """
    with numpy.load(file, **_allow_pickle_kwargs) as f:
        d = NpzDeserializer(
            f, path=path, strict=strict, ignore_names=ignore_names)
        d.load(obj)
