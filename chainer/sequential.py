import copy
import functools
import inspect

import chainer
from chainer import function


class Sequential(chainer.ChainList):

    """Sequential model which has a single-stream forward pass.

    This class enables to construct a network which has sequential structure
    easily. While :class:`~Chain` and :class:`~ChainList` can only take
    :class:`~Link` object as input to their constructor, this
    :class:`~Sequential` can take arbitrary number of any callable objects for
    the forward pass computation. A :class:`~Sequential` calls the given
    callable objects sequentially inside of the :meth:`~Sequential.__call__`
    method in the same order as the given argments.
    Therefore, you don't need to write the forward pass computation explicitly.

    .. admonition:: Example

        The below example code shows how to use this class to construct a
        simple sequential network::

          import chainer
          import chainer.functions as F
          import chainer.links as L
          from chainer import Sequential

          # Model definition without writing __call__ function
          model = Sequential(
              L.Linear(n_in, n_hidden),
              F.relu,
              L.Linear(n_hidden, n_hidden),
              F.relu,
              L.Linear(n_hidden, n_out)
          )

          # Compute the forward pass
          y = model(x)

        where ``x`` denotes a mini-batch of ``n_in``-dimensional input vectors.

        Furthermore, :class:`~Sequential` supports build-in list APIs, so you
        can concatenate :class:`~Sequential` objects and repeat a
        :class:`~Sequential` object to create a longer :class:`~Sequential`
        model easily with the same ways as Python lists::

          model_A = Sequential(L.Linear(10, 10), F.relu)
          model_B = Sequential(L.Linear(10, 10), F.sigmoid)
          model_C = model_A + model_B
          model_D = 3 * model_A

        You can also add your own functions or any callable objects to a
        :class:`~Sequential` object::

          from chainer.links.model.vision.vgg import VGG16Layers()

          model = Sequential()
          model.append(L.Linear(n_out, n_hidden))
          model.append(F.relu)
          model.append(F.Reshape((1, 3, 224, 224)))
          model.append(VGG16Layers())
          model.append(lambda x: x['prob'])

          y = model(x)

        The above code example shows how to add some layers to the ``model``
        using :meth:`~Sequential.append` method and then add a large network
        (``VGG16Layers``) and finally add a lambda function to extract the
        ``prob`` output.

    Args:
        layers: The layers which are called in its order. Each component should
            be a callable object including :class:`~Link` object and
            :class:`~Function` class.

    """

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self._layers = []
        self._n_lambda = 0
        for layer in layers:
            self.append(layer)

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    def __setitem__(self, i, layer):
        if i >= len(self):
            raise ValueError(
                '{} should be less than {}'.format(i, len(self)))
        if not callable(layer):
            raise ValueError(
                'All elements of a Sequential class should be callable. But '
                'given {} is not callable.'.format(layer))

        if self._layers[i] is not layer:
            del self[i]
            self.insert(i, layer)

    def __delitem__(self, i):
        layer = self._layers.pop(i)
        if isinstance(layer, chainer.Link):
            for i, link in enumerate(self._children):
                if link is layer:
                    del self._children[i]
                    break
        elif callable(layer) and hasattr(layer, '__name__') \
                and layer.__name__ == '<lambda>':
            self._n_lambda -= 1

    def __iter__(self):
        return iter(self._layers)

    def __reversed__(self):
        return reversed(self._layers)

    def __contains__(self, item):
        return item in self._layers

    def __add__(self, other):
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        elif callable(other):
            self.append(other)
            return self
        else:
            raise ValueError('add (+) operator is support only callable '
                             'objects, but {} is not callable'.format(
                                 str(type(other))))

    def __radd__(self, other):
        if callable(other):
            return Sequential(other) + self
        else:
            raise ValueError('add (+) operator is support only callable '
                             'objects, but {} is not callable'.format(
                                 str(type(other))))

    def __iadd__(self, other):
        if isinstance(other, Sequential):
            for layer in other:
                self.append(layer)
        elif callable(other):
            self.append(other)
        else:
            raise ValueError('add (+) operator is support only callable '
                             'objects, but {} is not callable'.format(
                                 str(type(other))))
        return self

    def __mul__(self, n_repeat):
        if n_repeat <= 0:
            return Sequential()
        ret = self.copy()
        for _ in range(n_repeat - 1):
            for layer in self:
                if isinstance(layer, chainer.Link):
                    layer = copy.deepcopy(layer)
                    for param in layer.params(include_uninit=False):
                        param.initialize(param.shape)
                else:
                    layer = copy.copy(layer)
                ret.append(layer)
        return ret

    def __rmul__(self, n_repeat):
        return self * n_repeat

    def __imul__(self, n_repeat):
        if n_repeat <= 0:
            self.clear()
            return self
        n_layers = len(self)
        for _ in range(n_repeat - 1):
            for i in range(n_layers):
                if isinstance(self[i], chainer.Link):
                    layer = copy.deepcopy(self[i])
                    for param in layer.params(include_uninit=False):
                        param.initialize(param.shape)
                else:
                    layer = copy.copy(layer)
                self.append(layer)
        return self

    def __call__(self, *x):
        """Forward pass computation.

        This method performs the forward pass computation by giving the input
        variable ``x`` to the layers registered in the constructor in the same
        order as the order in which the argments are given to the constructor.

        It should be noted that the input variable is given directly to the
        first layer and all intermediate outputs generated during the forward
        pass are also directly feeded to the next layer. Therefore, the number
        of output at a layer should be the same as the number of input at the
        next layer.

        Args:
            x: Input variables.

        Returns:
            The output of the final layer in the given layers.

        """
        for layer in self._layers:
            if isinstance(x, tuple):
                x = layer(*x)
            else:
                x = layer(x)
        return x

    def __reduce__(self):
        if self._n_lambda > 0:
            raise ValueError(
                'This Sequential object has at least one lambda function as '
                'its component. Lambda function can\'t be pickled, so please '
                'consider to use functools.partial instead of the lambda '
                'function or use "dill" which is an external package that '
                'enables pickling an object including lambda functions intead '
                'of built-in pickle.')
        return super(Sequential, self).__reduce__()

    def __str__(self):
        ret = ''
        for i, layer in enumerate(self):
            if isinstance(layer, Sequential):
                name = layer.__class__.__name__
                name += '\twhich has {} layers'.format(len(layer))
            elif isinstance(layer, chainer.Chain):
                name = layer.__class__.__name__
                name += '\tThe structure behind a Chain is determined at '
                name += 'runtime.'
            elif isinstance(layer, chainer.ChainList):
                name = layer.__class__.__name__
                name += '\tThe structure behind a ChainList is determined at '
                name += 'runtime.'
            elif isinstance(layer, chainer.Link):
                name = layer.__class__.__name__
                param_info = '\t'
                for param in sorted(layer.params(), key=lambda p: p.name):
                    param_info += param.name
                    if param._data[0] is not None:
                        param_info += str(param._data[0].shape)
                    else:
                        param_info += '(None)'
                    param_info += '\t'
                name = name + param_info
            elif isinstance(layer, function.Function):
                name = layer.__class__.__name__
            elif isinstance(layer, functools.partial):
                name = repr(layer)
            elif layer.__name__ == '<lambda>':
                name = inspect.getsource(layer).strip()
            else:
                name = layer.__name__
            ret += '{}\t{}\n'.format(i, name)
        return ret

    def append(self, layer):
        self.insert(len(self), layer)

    def extend(self, sequential):
        for layer in sequential:
            self.append(layer)

    def insert(self, i, layer):
        if not callable(layer):
            raise ValueError(
                'All elements of the argment should be callable. But '
                'given {} is not callable.'.format(layer))

        self._layers.insert(i, layer)
        if isinstance(layer, chainer.Link):
            self.add_link(layer)
        elif callable(layer) and hasattr(layer, '__name__') \
                and layer.__name__ == '<lambda>':
            self._n_lambda += 1

    def remove(self, layer):
        if layer in self:
            del self[self.index(layer)]
        else:
            raise ValueError(
                'There is no layer object that is same as {}'.format(layer))

    def remove_by_layer_type(self, type_name):
        """Remove layers by layer type.

        This method removes layers from the Sequential object by the
        layer's class name or function name. If you want to remove a
        :class:`~Link`, the argment ``type_name`` should be its class name,
        e.g., :class:`~links.Linear` or :class:`~links.Convolution2D`, etc.
        If you want to remove a :class:`~Function` class or any other callable
        objects, ``type_name`` should be the function name, e.g., ``relu`` or
        ``reshape``, etc.

        Args:
            type_name (str): The name of a layer you want to remove.

        """

        names = []
        for layer in self:
            if isinstance(layer, chainer.Link):
                name = layer.__class__.__name__
            else:
                name = layer.__name__
            names.append((name, layer))
        for _name, _layer in names:
            if type_name == _name:
                self.remove(_layer)

    def pop(self, i=-1):
        layer = self._layers[i]
        del self[i]
        return layer

    def clear(self):
        for i, link in enumerate(self._children):
            del self._children[i]
        self._layers = []

    def index(self, layer, start=None, end=None):
        return self._layers[start:end].index(layer)

    def count(self, layer):
        return self._layers.count(layer)

    def count_by_layer_type(self, type_name):
        """Count the number of layers by layer type.

        This method counts the number of layers which have the name given by
        the argment ``type_name``. For example, if you want to know the number
        of :class:`~links.Linear` layers included in this model, ``type_name``
        should be ``Linear``. If you want to know the number of
        :class:`~Function` classes or user-defined functions which have a
        specific name, ``type_name`` should be the function name, e.g.,
        ``relu`` or ``reshape``, etc.

        Args:
            type_name (str): The class or function name of a layer you want to
                enumerate.

        """

        num = 0
        for layer in self._layers:
            if isinstance(layer, chainer.Link):
                if layer.__class__.__name__ == type_name:
                    num += 1
            else:
                if layer.__name__ == type_name:
                    num += 1
        return num

    def sort(self, key=None, reverse=None):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError

    def copy(self):
        ret = Sequential()
        ret._n_lambda = self._n_lambda
        for layer in self:
            if isinstance(layer, chainer.Link):
                ret.append(layer.copy())
            else:
                ret.append(copy.copy(layer))
        return ret
