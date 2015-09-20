import copy as copy_module
import sys

import numpy
import six

from chainer import cuda


class ParameterizedObject(object):

    def __init__(self, name=''):
        self.params = {}
        self.states = {}
        self._name = name
        self._volatile = False

    @property
    def name(self):
        return self._name or '/'

    @name.setter
    def name(self, name):
        self._name = name if name != '/' else ''

    @property
    def volatile(self):
        return self._volatile

    @volatile.setter
    def volatile(self, value):
        value = bool(value)
        if self._volatile is value:
            return
        for _, param in self.visitparams():
            param.volatile = value
        self._volatile = value

    def copy(self, shared=True):
        ret = copy_module.copy(self)

        copy = copy_module.copy if shared else copy_module.deepcopy
        ret.params = {}
        for key, param in six.iteritems(self.params):
            ret.params[key] = copy(param)
        ret.states = copy(self.states)
        return ret

    def to_cpu(self):
        for obj in self.visithierarchy():
            for param in six.itervalues(obj.params):
                param.data = cuda.to_cpu(param.data)
                param._grad = cuda.to_cpu(param._grad)

            states = obj.states
            for key, value in six.iteritems(states):
                states[key] = cuda.to_cpu(value)

        return self

    def to_gpu(self, device=None):
        cupy = cuda.cupy
        with cuda.get_device(device):
            for obj in self.visithierarchy():
                for param in six.itervalues(obj.params):
                    param.data = cupy.asarray(param.data)
                    if param._grad is not None:
                        param._grad = cupy.asarray(param._grad)

                states = obj.states
                for key, value in six.iteritems(states):
                    states[key] = cupy.asarray(value)

        return self

    def visitparams(self):
        for obj in self.visithierarchy():
            prefix = obj._name + '/_params/'
            for key, param in six.iteritems(obj.params):
                yield prefix + key, param

    def visithierarchy(self):
        yield self

    def copyparams(self, obj):
        params = {}
        for path, param in obj.visitparams():
            params[path] = param.data
        for path, param in self.visitparams():
            dst = param.data
            src = params[path]
            if isinstance(dst, numpy.ndarray):
                numpy.copyto(dst, cuda.to_cpu(src))
            elif isinstance(src, numpy.ndarray):
                dst.set(src)
            else:
                cuda.cupy.copyto(dst, src)

    def zerograds(self):
        for obj in self.visithierarchy():
            params = obj.params
            for key, p in six.iteritems(params):
                arr = p._grad
                if arr is None:
                    data = p.data
                    xp = cuda.get_array_module(data)
                    p._grad = xp.zeros_like(data)
                else:
                    arr.fill(0)

    def addgrads(self, obj):
        grads = {}
        for path, param in obj.visitparams():
            grads[path] = param._grad
        for path, param in self.visitparams():
            dst = param._grad
            src = grads[path]
            if isinstance(dst, numpy.ndarray):
                dst += cuda.to_cpu(src)
            elif isinstance(src, numpy.ndarray):
                dst += cuda.to_gpu(src, device=dst)
            elif src.device == dst.device:
                dst += src
            else:
                dst += cuda.copy(src, out_device=dst)

    def serialize(self, serializer):
        p = serializer['_params']
        for key, param in six.iteritems(self.params):
            param.data = p(key, param.data)
            # grad is not serialized

        states = self.states
        s = serializer['_states']
        for key, state in six.iteritems(states):
            states[key] = s(key, state)


class ParameterizedDict(ParameterizedObject):

    def __init__(self, **kwds):
        ParameterizedObject.__init__(self)
        self.children = kwds

        prefix = self._name + '/'
        for key, obj in six.iteritems(kwds):
            if not isinstance(obj, ParameterizedObject):
                raise TypeError('Cannot set a non-parameterized object to '
                                'ParameterizedDict')
            if obj._name:
                raise ValueError('Cannot set a parameterized object to '
                                 'multiple parents')
            obj.name = prefix + key

    def __contains__(self, key):
        return key in self.children

    def __delitem__(self, key):
        self.children[key].name = ''
        del self.children[key]

    def __iter__(self):
        return self.children.__iter__()

    def __getitem__(self, key):
        return self.children[key]

    def __len__(self):
        return len(self.children)

    def __setitem__(self, key, value):
        if not isinstance(value, ParameterizedObject):
            raise TypeError('Cannot set a non-parameterized object to '
                            'ParameterizedDict')
        if value._name:
            raise ValueError('Cannot set a parameterized object to multiple '
                             'parents')
        value.name = '%s/%s' % (self._name, key)

        old = self.get(key, None)
        if old is not None:
            old.name = ''
        self.children[key] = value

    def clear(self):
        for obj in six.itervalues(self.children):
            obj.name = ''
        self.children.clear()

    def get(self, key, *args):
        return self.children.get(key, *args)

    def items(self):
        return self.children.items()

    if sys.version_info.major < 3:
        def iteritems(self):
            return self.children.iteritems()

        def iterkeys(self):
            return self.children.iterkeys()

        def itervalues(self):
            return self.children.itervalues()

    def has_key(self, key):
        return key in self

    def keys(self):
        return self.children.keys()

    def pop(self, key, *args):
        ret = self.children.pop(key, *args)
        if args and ret is args[0]:
            return ret
        ret.name = ''
        return ret

    def popitem(self):
        key, obj = self.children.popitem()
        obj.name = ''
        return key, obj

    def setdefault(self, key, default=None):
        ret = self.children.get(key, None)
        if ret is None:
            self[key] = default
            return default
        else:
            return ret

    def values(self):
        return self.children.values()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name == '/':
            name = ''
        self._name = name
        prefix = self._name + '/'
        for key, obj in six.iteritems(self):
            obj.name = prefix + key

    def copy(self, shared=True):
        ret = ParameterizedObject.copy(self, shared)
        for key, obj in six.iteritems(self):
            ret[key] = obj.copy(shared)
        return ret

    def visithierarchy(self):
        yield self
        for o1 in six.itervalues(self):
            for o2 in o1.visithierarchy():
                yield o2

    def serialize(self, serializer):
        ParameterizedObject.serialize(self, serializer)
        for key, obj in six.iteritems(self):
            obj.serialize(serializer[key])


class ParameterizedList(ParameterizedObject):

    def __init__(self, *args):
        ParameterizedObject.__init__(self)
        for obj in args:
            if not isinstance(obj, ParameterizedObject):
                raise TypeError('Cannot set a non-parameterized object to '
                                'ParameterizedList')
            if obj._name:
                raise ValueError('Cannot set a parameterized object to '
                                 'multiple parents')
        for i, obj in enumerate(args):
            obj.name = '%s/%d' % (self._name, i)
        self.children = list(args)

    def __getitem__(self, idx):
        return self.children[idx]

    def __iter__(self):
        return self.children.__iter__()

    def __len__(self):
        return len(self.children)

    def __setitem__(self, idx, value):
        if not isinstance(value, ParameterizedObject):
            raise TypeError('Cannot set a non-parameterized object to '
                            'ParamterizedList')
        if value._name:
            raise ValueError('Cannot set a parameterized object to multiple '
                             'parents')
        value.name = '%s/%d' % (self._name, idx)

        self.children[idx].name = ''
        self.children[idx] = value

    def append(self, obj):
        if not isinstance(obj, ParameterizedObject):
            raise TypeError('Cannot set a non-parameterized object to '
                            'ParameterizedList')
        if obj._name:
            raise ValueError('Cannot set a parameterized object to multiple '
                             'parents')
        obj.name = '%s/%d' % (self._name, len(self.children))
        self.children.append(obj)

    def pop(self):
        obj = self.children.pop()
        obj.name = ''
        return obj

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        for i, obj in enumerate(self):
            obj.name = '%s/%d' % (name, i)

    def copy(self, shared=True):
        ret = ParameterizedObject.copy(self, shared)
        for i, obj in enumerate(self):
            ret[i] = obj.copy(shared)
        return ret

    def visithierarchy(self):
        yield self
        for o1 in self:
            for o2 in o1.visithierarchy():
                yield o2

    def serialize(self, serializer):
        ParameterizedObject.serialize(self, serializer)
        for idx, obj in enumerate(self):
            obj.serialize(serializer[str(idx)])


def _apply_on_variable(v, func):
    v.data = func(v.data)
    if v._grad is not None:
        v._grad = func(v._grad)
