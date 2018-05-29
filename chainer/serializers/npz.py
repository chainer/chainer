import numpy
import six

from chainer.backends import cuda
from chainer import serializer
from chainer.utils import argument


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
        elif isinstance(value, numpy.ndarray):
            numpy.copyto(value, dataset)
        elif isinstance(value, cuda.ndarray):
            value.set(numpy.asarray(dataset))
        else:
            value = type(value)(numpy.asarray(dataset))
        return value


def load_npz(file, obj, path='', strict=True, **kwargs):
    """load_npz(file, obj, path='', strict=True, *, allow_pickle=None)

    Loads an object from the file in NPZ format.

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
        allow_pickle (bool): If ``True``, the deserializer allows loading
            pickled object arrays. The default is ``False`` for security
            reasons (see :func:`numpy.load` for details).

    .. note::
        If you are using NumPy 1.9, ``allow_pickle`` option cannot be specified
        and load of pickled object arrays is **always allowed**.

    .. seealso::
        :func:`chainer.serializers.save_npz`

    """
    allow_pickle, = argument.parse_kwargs(kwargs, ('allow_pickle', None))

    if numpy.lib.NumpyVersion(numpy.__version__) >= '1.10.0':
        if allow_pickle is None:
            allow_pickle = False
        load_kwargs = {'allow_pickle': allow_pickle}
    else:
        if allow_pickle is not None:
            raise ValueError(
                'NumPy 1.10 or later is required to use allow_pickle option')
        load_kwargs = {}

    with numpy.load(file, **load_kwargs) as f:
        d = NpzDeserializer(f, path=path, strict=strict)
        d.load(obj)
