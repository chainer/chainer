import copy
import inspect

from chainer import link as _link


class Sequential(_link.ChainList):

    """Sequential model which has a single-stream forward pass.

    .. warning::

        This feature is experimental. The interface can change in the future.

    This class enables to construct a network which has sequential structure
    easily. While :class:`~chainer.Chain` and :class:`~chainer.ChainList` can
    only take :class:`~chainer.Link` object as input to their constructor, this
    :class:`Sequential` can take arbitrary number of any callable objects for
    the forward pass computation. A :class:`Sequential` calls the given
    callable objects sequentially inside of the :meth:`~Sequential.forward`
    method in the same order as the given arguments.
    Therefore, you do not need to write the forward pass computation
    explicitly.

    .. admonition:: Example

        The below example code shows how to use this class to construct a
        simple sequential network::

          import chainer
          import chainer.functions as F
          import chainer.links as L
          from chainer import Sequential

          # Model definition without writing forward function
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

        Furthermore, :class:`Sequential` supports built-in list APIs, so you
        can concatenate :class:`Sequential` objects to create a longer
        :class:`Sequential` model easily with the same ways as Python lists:

        >>> from chainer import Sequential
        >>> model_A = Sequential(L.Linear(10, 10), F.relu)
        >>> model_B = Sequential(L.Linear(10, 10), F.sigmoid)
        >>> model_C = model_A + model_B

        To repeat a :class:`Sequential` object multiple times, you can use
        :meth:`~chainer.Link.repeat` method.

        >>> model_D = model_A.repeat(3)

        You can also add your own functions or any callable objects to a
        :class:`Sequential` object::

          from chainer.links.model.vision.vgg import VGG16Layers

          model = Sequential()
          model.append(L.Linear(n_out, n_hidden))
          model.append(F.relu)
          model.append(lambda x: F.reshape(x, (1, 3, 224, 224)))
          model.append(VGG16Layers())
          model.append(lambda x: x['prob'])

          y = model(x)

        The above code example shows how to add some layers to the ``model``
        using :meth:`~Sequential.append` method and then add a large network
        (``VGG16Layers``) and finally add a lambda function to extract the
        ``prob`` output.

        You can check the structure of your model briefly using ``print``
        as following:

        >>> print(model_C)  # doctest: +ELLIPSIS
        Sequential(
          (0): Linear(in_size=10, out_size=10, nobias=False),
          (1): <function relu at 0x...>,
          (2): Linear(in_size=10, out_size=10, nobias=False),
          (3): <function sigmoid at 0x...>,
        )

        .. note::

            Note that a :class:`Sequential` link which has at least one
            ``lambda`` function as its member cannot be pickled. So, please
            use ``partial`` method from :mod:`functools` package instead::

              from functools import partial

              # This is not pickable
              model = Sequential(
                  L.Convolution2D(None, 64, 3, 1, 1),
                  lambda x: F.max_pooling_2d(x, 2)
              )

              # This is pickable
              model = Sequential(
                  L.Convolution2D(None, 64, 3, 1, 1),
                  partial(F.max_pooling_2d, ksize=2)
              )

    Args:
        layers: The layers which are called in its order. Each component should
            be a callable object including :class:`~chainer.Link` object and
            functions defined under the :mod:`chainer.functions`, e.g.,
            :func:`~chainer.functions.relu`, etc.

    """

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self._layers = []
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
        if isinstance(layer, _link.Link):
            for i, link in enumerate(self._children):
                if link.name == layer.name:
                    del self._children[i]
                    break
            for j, layer in enumerate(self._children[i:]):
                layer.name = str(i + j)

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
        else:
            raise ValueError('add (+) operator supports only objects of '
                             'Sequential class, but {} is given.'.format(
                                 str(type(other))))

    def __iadd__(self, other):
        if isinstance(other, Sequential):
            for layer in other:
                self.append(layer)
        else:
            raise ValueError('add (+) operator supports only objects of '
                             'Sequential class, but {} is given.'.format(
                                 str(type(other))))
        return self

    def forward(self, *x):
        """Forward pass computation.

        This method performs the forward pass computation by giving the input
        variable ``x`` to the layers registered in the constructor in the same
        order as the order in which the arguments are given to the constructor.

        It should be noted that the input variable is given directly to the
        first layer and all intermediate outputs generated during the forward
        pass are also directly fed to the next layer. Therefore, the number
        of outputs at a layer should be the same as the number of inputs at
        the next layer.

        Args:
            x: Input variables.

        Returns:
            The output of the final layer in the given layers.

        """
        if not self._layers:
            raise RuntimeError('Sequential does not have any layer.')

        for layer in self._layers:
            if isinstance(x, tuple):
                x = layer(*x)
            else:
                x = layer(x)
        return x

    def __reduce__(self):
        n_lambda = 0
        for layer in self._layers:
            if callable(layer) and hasattr(layer, '__name__') \
                    and layer.__name__ == '<lambda>':
                n_lambda += 1

        if n_lambda > 0:
            raise ValueError(
                'This Sequential object has at least one lambda function as '
                'its component. Lambda function can not be pickled, so please '
                'consider to use functools.partial instead of the lambda '
                'function or use "dill", which is an external package that '
                'enables pickling an object including lambda functions intead '
                'of built-in pickle.')
        return super(Sequential, self).__reduce__()

    def __str__(self):
        reps = []
        for index, layer in enumerate(self):
            # Explore better representation by if-block.
            if getattr(layer, '__name__', None) == '<lambda>':
                rep = inspect.getsource(layer).strip().rstrip(',')
            else:
                rep = str(layer)
            # Add indentation to each line.
            rep = '({index}): {rep},'.format(index=index, rep=rep)
            for line in rep.splitlines():
                reps.append('  {line}\n'.format(line=line))
        reps = ''.join(reps)
        if reps:  # No newline with no layers.
            reps = '\n' + reps

        return '{cls}({layers})'.format(
            cls=self.__class__.__name__, layers=reps,
        )

    def append(self, layer):
        self.insert(len(self), layer)

    def extend(self, sequential):
        for layer in sequential:
            self.append(layer)

    def insert(self, i, layer):
        if not callable(layer):
            raise ValueError(
                'All elements of the argument should be callable. But '
                'given {} is not callable.'.format(layer))

        self._layers.insert(i, layer)
        if isinstance(layer, _link.Link):
            if i == 0:
                self._children.insert(0, layer)
            else:
                if i < 0:
                    i = len(self._layers) + i
                last_link_pos = 0
                for j in range(i - 1, -1, -1):
                    # The last link before the given position
                    if isinstance(self._layers[j], _link.Link):
                        last_link_pos = j
                self._children.insert(last_link_pos + 1, layer)
            for i, layer in enumerate(self._children):
                layer.name = str(i)

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
        :class:`~Link`, the argument ``type_name`` should be its class name,
        e.g., :class:`~links.Linear` or :class:`~links.Convolution2D`, etc.
        If you want to remove a :class:`~Function` class or any other callable
        objects, ``type_name`` should be the function name, e.g., ``relu`` or
        ``reshape``, etc.

        Args:
            type_name (str): The name of a layer you want to remove.

        """

        names = []
        for layer in self:
            if isinstance(layer, _link.Link):
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
        # TODO(mitmul): Reduce the computational cost here
        for i, _ in enumerate(self._children):
            del self._children[i]
        self._layers = []

    def index(self, layer, start=None, end=None):
        return self._layers[start:end].index(layer)

    def count(self, layer):
        return self._layers.count(layer)

    def count_by_layer_type(self, type_name):
        """Count the number of layers by layer type.

        This method counts the number of layers which have the name given by
        the argument ``type_name``. For example, if you want to know the number
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
            if isinstance(layer, _link.Link):
                if layer.__class__.__name__ == type_name:
                    num += 1
            else:
                if layer.__name__ == type_name:
                    num += 1
        return num

    def copy(self, mode='share'):
        ret = Sequential()
        for layer in self:
            if isinstance(layer, _link.Link):
                ret.append(layer.copy(mode))
            else:
                ret.append(copy.copy(layer))
        return ret

    def copyparams(self, link, copy_persistent=True):
        if not isinstance(link, Sequential):
            raise ValueError('Objects other than Sequential object cannot be '
                             'copied to a Sequential object.')
        for idx, child in enumerate(self):
            if isinstance(child, _link.Link):
                child.copyparams(link[idx], copy_persistent)

    def flatten(self):
        """Flatten nested :class:`~chainer.Sequential` links.

        This method flattens all the nested :class:`~chainer.Sequential` links
        inside this :class:`~chainer.Sequential` link.

        Returns:

            A flattened :class:`~chainer.Sequential` object.

        .. admonition:: Example

            .. code-block:: python

                >>> import chainer
                >>> import chainer.functions as F
                >>> import chainer.links as L
                >>> a = chainer.Sequential(L.Linear(None, 10), F.relu)
                >>> b = chainer.Sequential(L.Linear(None, 10), F.relu)
                >>> a.append(b)
                >>> print(a)  # Without flatten
                0       Linear  W(None) b(10,)
                1       relu
                2       Sequential      which has 2 layers
                >>> print(a.flatten())  # With flatten
                0       Linear  W(None) b(10,)
                1       relu
                2       Linear  W(None) b(10,)
                3       relu

        """
        ret = Sequential()
        for layer in self:
            if isinstance(layer, Sequential):
                ret.extend(layer.flatten())
            else:
                ret.append(layer)
        return ret
