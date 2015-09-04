import copy as copy_module
import sys

import numpy
import six

from chainer import cuda


class Model(object):

    def __init__(self, name=''):
        self.params = {}
        self.grads = {}
        self.states = {}
        self._name = name

    @property
    def name(self):
        return self._name or '/'

    @name.setter
    def name(self, name):
        self._name = name if name != '/' else ''

    def copy(self, shared=True):
        ret = copy_module.copy(self)

        copy = copy_module.copy if shared else copy_module.deepcopy
        ret.params = copy(self.params)
        ret.grads = copy(self.grads)
        ret.states = copy(self.states)
        return ret

    def to_cpu(self):
        for model in self.visitmodels():
            _dict_to_cpu(model.params)
            _dict_to_cpu(model.grads)
            _dict_to_cpu(model.states)

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            for model in self.visitmodels():
                _dict_to_gpu(model.params)
                _dict_to_gpu(model.grads)
                _dict_to_gpu(model.states)

    def visitparams(self, silent=False):
        for model in self.visitmodels():
            prefix = model._name + '/_params/'
            p, g = model.params, model.grads
            for key in p:
                grad = g.get(key, None) if silent else g[key]
                yield prefix + key, p[key], grad

    def visitmodels(self):
        yield self

    def copyparams(self, model):
        params = {}
        for path, param, _ in model.visitparams(True):
            params[path] = param
        for path, dst, _ in self.visitparams(True):
            src = params[path]
            if isinstance(dst, numpy.ndarray):
                numpy.copyto(dst, cuda.to_cpu(src))
            elif isinstance(src, numpy.ndarray):
                dst.set(src)
            else:
                cuda.cupy.copyto(dst, src)

    def zerograds(self):
        for model in self.visitmodels():
            params, grads = model.params, model.grads
            for key, p in six.iteritems(params):
                arr = grads.get(key, None)
                if arr is None:
                    xp = cuda.get_array_module(p)
                    arr = xp.zeros_like(p)
                    grads[key] = arr
                else:
                    arr.fill(0)

    def addgrads(self, model):
        grads = {}
        for path, _, grad in model.visitparams():
            grads[path] = grad
        for path, _, dst in self.visitparams():
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
        _serialize_dict(serializer['_params'], self.params)
        _serialize_dict(serializer['_states'], self.states)


class ModelDict(Model):

    def __init__(self, **kwds):
        Model.__init__(self)
        self.models = kwds

        prefix = self._name + '/'
        for key, model in six.iteritems(kwds):
            if not isinstance(model, Model):
                raise TypeError('Cannot set a non-model object to ModelDict')
            if model._name:
                raise ValueError('Cannot set a model to multiple parents')
            model.name = prefix + key

    def __contains__(self, key):
        return key in self.models

    def __delitem__(self, key):
        self.models[key].name = ''
        del self.models[key]

    def __iter__(self):
        return self.models.__iter__()

    def __getitem__(self, key):
        return self.models[key]

    def __len__(self):
        return len(self.models)

    def __setitem__(self, key, value):
        if not isinstance(value, Model):
            raise TypeError('Cannot set a non-model object to ModelDict')
        if value._name:
            raise ValueError('Cannot set a model to multiple parents')
        value.name = '%s/%s' % (self._name, key)

        old = self.get(key, None)
        if old is not None:
            old.name = ''
        self.models[key] = value

    def clear(self):
        for model in six.itervalues(self.models):
            model.name = ''
        self.models.clear()

    def get(self, key, *args):
        return self.models.get(key, *args)

    def items(self):
        return self.models.items()

    if sys.version_info.major < 3:
        def iteritems(self):
            return self.models.iteritems()

        def iterkeys(self):
            return self.models.iterkeys()

        def itervalues(self):
            return self.models.itervalues()

    def has_key(self, key):
        return key in self

    def keys(self):
        return self.models.keys()

    def pop(self, key, *args):
        ret = self.models.pop(key, *args)
        if args and ret is args[0]:
            return ret
        ret.name = ''
        return ret

    def popitem(self):
        key, model = self.models.popitem()
        model.name = ''
        return key, model

    def setdefault(self, key, default=None):
        ret = self.models.get(key, None)
        if ret is None:
            self[key] = default
            return default
        else:
            return ret

    def values(self):
        return self.models.values()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name == '/':
            name = ''
        self._name = name
        prefix = self._name + '/'
        for key, model in six.iteritems(self):
            model.name = prefix + key

    def copy(self, shared=True):
        ret = Model.copy(self, shared)
        for key, model in six.iteritems(self):
            ret[key] = model.copy(shared)
        return ret

    def visitmodels(self):
        yield self
        for m1 in six.itervalues(self):
            for m2 in m1.visitmodels():
                yield m2

    def serialize(self, serializer):
        Model.serialize(self, serializer)
        for key, model in six.iteritems(self):
            model.serialize(self, serializer[key])


class ModelList(Model):

    def __init__(self, *args):
        Model.__init__(self)
        for obj in args:
            if not isinstance(obj, Model):
                raise TypeError('Cannot set a non-model object to ModelList')
            if obj._name:
                raise ValueError('Cannot set a model to multiple parents')
        for i, model in enumerate(args):
            model.name = '%s/%d' % (self._name, i)
        self.models = list(args)

    def __getitem__(self, idx):
        return self.models[idx]

    def __iter__(self):
        return self.models.__iter__()

    def __len__(self):
        return len(self.models)

    def __setitem__(self, idx, value):
        if not isinstance(value, Model):
            raise TypeError('Cannot set a non-model object to ModelList')
        if value._name:
            raise ValueError('Cannot set a model to multiple parents')
        value.name = '%s/%d' % (self._name, idx)

        self.models[idx].name = ''
        self.models[idx] = value

    def append(self, model):
        if not isinstance(model, Model):
            raise TypeError('Cannot set a non-model object to ModelList')
        if model._name:
            raise ValueError('Cannot set a model to multiple parents')
        model.name = '%s/%d' % (self._name, len(self.models))
        self.models.append(model)

    def pop(self):
        model = self.models.pop()
        model.name = ''
        return model

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        for i, model in enumerate(self):
            model.name = '%s/%d' % (name, i)

    def copy(self, shared=True):
        ret = Model.copy(self, shared)
        for i, model in enumerate(self):
            ret[i] = model.copy(shared)
        return ret

    def visitmodels(self):
        yield self
        for m1 in self:
            for m2 in m1.visitmodels():
                yield m2

    def serialize(self, serializer):
        Model.save(self, serializer)
        for idx, model in six.iteritems(self):
            model.save(self, serializer[str(idx)])


def _dict_to_cpu(d):
    for key in d:
        d[key] = cuda.to_cpu(d[key])


def _dict_to_gpu(d):
    for key in d:
        value = d[key]
        if isinstance(value, numpy.ndarray):
            d[key] = cuda.to_gpu(value)


def _serialize_dict(s, d):
    for key in d:
        d[key] = s(key, d[key])
