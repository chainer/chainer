import collections
import contextlib
import copy
import warnings

import numpy
import six

from chainer import cuda
from chainer import initializers
from chainer import variable


def _is_shape(value):
    if value is None:
        return True
    elif isinstance(value, collections.Sequence):
        try:
            return all(int(x) for x in value)
        except TypeError:
            return False
    try:
        return int(value)
    except TypeError:
        return False


def _ensure_shape_dtype(value):
    # Return value paired with dtype FP32 if it is a shape.
    if _is_shape(value):
        return value, 'f'
    # Otherwise, returns it with assuming a shape-dtype pair.
    else:
        return value


class Link(object):

    """Building block of model definitions.

    Link is a building block of neural network models that support various
    features like handling parameters, defining network fragments,
    serialization, etc.

    Link is the primitive structure for the model definitions. It supports
    management of parameter variables and *persistent values* that should be
    incorporated to serialization.

    Parameter is an instance of :class:`~chainer.Parameter` registered to a
    link. A :class:`~chainer.Parameter` object can be registered as a
    parameter of the link by assigning it to an attribute within *an
    initialization scope*, which is a code surrounded by a
    :meth:`init_scope` context manager using the ``with`` statement.

    Persistent values are arrays, scalars, or any other serializable values
    registered via :meth:`register_persistent` or :meth:`add_persistent`.

    .. note::
       Whereas arbitrary serializable objects can be registered as persistent
       values, it is strongly recommended to just register values that should
       be treated as results of learning. A typical example of persistent
       values is ones computed during training and required for testing, e.g.
       running statistics for batch normalization.

    Parameters and persistent values are referred by their names. They can be
    accessed as attributes of the links. Link class itself manages the lists
    of names of parameters and persistent values to distinguish parameters and
    persistent values from other attributes.

    Link can be composed into more complex models. This composition feature is
    supported by child classes like :class:`Chain` and :class:`ChainList`. One
    can create a chain by combining one or more links. See the documents for
    these classes for details.

    As noted above, Link supports the serialization protocol of the
    :class:`~chainer.Serializer` class. **Note that only parameters and
    persistent values are saved and loaded.** Other attributes are considered
    as a part of user program (i.e. a part of network definition). In order to
    construct a link from saved file, other attributes must be identically
    reconstructed by user codes.

    .. admonition:: Example

       This is a simple example of custom link definition. Chainer itself also
       provides many links defined under the :mod:`~chainer.links` module. They
       might serve as examples, too.

       Consider we want to define a simple primitive link that implements a
       fully-connected layer based on the :func:`~functions.linear` function.
       Note that this function takes input units, a weight variable, and a bias
       variable as arguments. Then, the fully-connected layer can be defined as
       follows::

          import chainer
          import chainer.functions as F
          from chainer import initializers
          import numpy as np

          class LinearLayer(chainer.Link):

              def __init__(self, n_in, n_out):
                  super(LinearLayer, self).__init__()
                  with self.init_scope():
                      self.W = chainer.Parameter(
                          (n_out, n_in), initializers.Normal())
                      self.b = chainer.Parameter(
                          (n_out,), initializers.Zero())

              def __call__(self, x):
                  return F.linear(x, self.W, self.b)

       This example shows that a user can define arbitrary parameters and use
       them in any methods. Links typically implement the ``__call__``
       operator, although they can also provide other methods to implement the
       forward propagation.

    Args:
        params: *(deprecated since v2.0.0)* Names, shapes, and optional dtypes
            of initial parameters. The keywords are used as the parameter
            names and the corresponding values consist either of the shape or
            a tuple of shape and a dtype ``(shape, dtype)``. If only the shape
            is supplied, the default dtype will be used.

    Attributes:
        name (str): Name of this link, given by the parent chain (if exists).

    """

    def __init__(self, **params):
        self._params = set()
        self._persistent = set()
        self._cpu = True
        self._device_id = None
        self._within_init_scope = False
        self.name = None

        for name, value in six.iteritems(params):
            # Note: deprecation warning will be raised in add_param
            shape, dtype = _ensure_shape_dtype(value)
            self.add_param(name, shape, dtype=dtype)

    @property
    def xp(self):
        """Array module for this link.

        Depending on which of CPU/GPU this link is on, this property returns
        :mod:`numpy` or :mod:`cupy`.

        """
        return numpy if self._cpu else cuda.cupy

    @property
    def within_init_scope(self):
        """True if the current code is inside of an initialization scope.

        See :meth:`init_scope` for the details of the initialization scope.

        """
        return getattr(self, '_within_init_scope', False)

    @contextlib.contextmanager
    def init_scope(self):
        """Creates an initialization scope.

        This method returns a context manager object that enables registration
        of parameters (and links for :class:`~chainer.Chain`) by an assignment.
        A :class:`~chainer.Parameter` object can be automatically registered
        by assigning it to an attribute under this context manager.

        .. admonition:: Example

           In most cases, the parameter registration is done in the
           initializer method. Using the ``init_scope`` method, we can
           simply assign a :class:`~chainer.Parameter` object to register
           it to the link.

           .. code-block:: python

              class MyLink(chainer.Link):
                  def __init__(self):
                      super().__init__()
                      with self.init_scope():
                          self.W = chainer.Parameter(0, (10, 5))
                          self.b = chainer.Parameter(0, (5,))

        """
        old_flag = self.within_init_scope
        self._within_init_scope = True
        try:
            yield
        finally:
            self._within_init_scope = old_flag

    def __setattr__(self, name, value):
        if self.within_init_scope and isinstance(value, variable.Parameter):
            value.name = name
            if not self._cpu:
                value.to_gpu(self._device_id)
            self._params.add(name)
            self._persistent.discard(name)
        super(Link, self).__setattr__(name, value)

    def __delattr__(self, name):
        self._params.discard(name)
        self._persistent.discard(name)
        super(Link, self).__delattr__(name)

    def add_param(self, name, shape=None, dtype=numpy.float32,
                  initializer=None):
        """Registers a parameter to the link.

        .. deprecated:: v2.0.0

           Assign a :class:`~chainer.Parameter` object directly to an
           attribute within :meth:`an initialization scope <init_scope>`
           instead. For example, the following code

           .. code-block:: python

               link.add_param('W', shape=(5, 3))

           can be replaced by the following assignment.

           .. code-block:: python

               with self.init_scope():
                   link.W = chainer.Parameter(None, (5, 3))

           The latter one is easier for IDEs to keep track of the attribute's
           type.

        Args:
            name (str): Name of the parameter. This name is also used as the
                attribute name.
            shape (int or tuple of ints): Shape of the parameter array. If it
                is omitted, the parameter variable is left uninitialized.
            dtype: Data type of the parameter array.
            initializer: If it is not ``None``, the data is initialized with
                the given initializer. If it is an array, the data is directly
                initialized by it. If it is callable, it is used as a weight
                initializer. Note that in these cases, ``dtype`` argument is
                ignored.

        """
        warnings.warn('''\
Parameter registeration via Link.__init__ and Link.add_param are deprecated.
Assign a Parameter object directly to an attribute within a \
"with Link.init_scope():" block instead.
''', DeprecationWarning)
        if name in self.__dict__:
            raise AttributeError(
                'cannot register a new parameter %s: attribute exists'
                % name)
        if initializer is None:
            initializer = initializers.NaN(dtype)
        param = variable.Parameter(initializer, shape)
        with self.init_scope():
            setattr(self, name, param)

    def add_persistent(self, name, value):
        """Registers a persistent value to the link.

        The registered value is saved and loaded on serialization and
        deserialization. The value is set to an attribute of the link.

        Args:
            name (str): Name of the persistent value. This name is also used
                for the attribute name.
            value: Value to be registered.

        """
        d = self.__dict__
        if name in d:
            raise AttributeError(
                'cannot register a new persistent value %s: attribute exists'
                % name)
        self._persistent.add(name)
        self._params.discard(name)
        d[name] = value

    def register_persistent(self, name):
        """Registers an attribute of a given name as a persistent value.

        This is a convenient method to register an existing attribute as a
        persistent value. If ``name`` has been already registered as a
        parameter, this method removes it from the list of parameter names
        and re-registers it as a persistent value.

        Args:
            name (str): Name of the attribute to be registered.

        """
        if not hasattr(self, name):
            raise AttributeError(
                'cannot register non-existent attribute %s as a persistent '
                'value' % name)
        self._persistent.add(name)
        self._params.discard(name)

    def copy(self):
        """Copies the link hierarchy to new one.

        The whole hierarchy rooted by this link is copied. The copy is
        basically shallow, except that the parameter variables are also
        shallowly copied. It means that the parameter variables of copied one
        are different from ones of original link, while they share the data and
        gradient arrays.

        The name of the link is reset on the copy, since the copied instance
        does not belong to the original parent chain (even if exists).

        Returns:
            Link: Copied link object.

        """
        ret = copy.copy(self)
        ret._params = set(self._params)
        ret._persistent = set(self._persistent)
        ret.name = None
        d = ret.__dict__
        for name in ret._params:
            d[name] = copy.copy(d[name])
            d[name].grad = None
        return ret

    def to_cpu(self):
        """Copies parameter variables and persistent values to CPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to CPU, the link implementation must
        override this method to do so.

        Returns: self

        """
        if self._cpu:
            return self
        d = self.__dict__
        for name in self._params:
            d[name].to_cpu()
        for name in self._persistent:
            value = d[name]
            if isinstance(value, cuda.ndarray):
                d[name] = value.get()
        self._cpu = True
        self._device_id = None
        return self

    def to_gpu(self, device=None):
        """Copies parameter variables and persistent values to GPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to GPU, the link implementation must
        override this method to do so.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        Returns: self

        """
        cuda.check_cuda_available()
        if not self._cpu:
            return self
        d = self.__dict__
        with cuda._get_device(device):
            for name in self._params:
                d[name].to_gpu()
            for name in self._persistent:
                value = d[name]
                if isinstance(value, numpy.ndarray):
                    d[name] = cuda.to_gpu(value)
            self._device_id = cuda.cupy.cuda.get_device_id()
        self._cpu = False
        return self

    def params(self, include_uninit=True):
        """Returns a generator of all parameters under the link hierarchy.

        Args:
            include_uninit (bool): If ``True``, it also generates uninitialized
                parameters.

        Returns:
            A generator object that generates all parameters.

        """
        d = self.__dict__
        for name in self._params:
            if include_uninit or d[name].data is not None:
                yield d[name]

    def namedparams(self, include_uninit=True):
        """Returns a generator of all (path, param) pairs under the hierarchy.

        Args:
            include_uninit (bool): If ``True``, it also generates uninitialized
                parameters.

        Returns:
            A generator object that generates all (path, parameter) pairs. The
            paths are relative from this link.

        """
        d = self.__dict__
        for name in self._params:
            if include_uninit or d[name].data is not None:
                yield '/' + name, d[name]

    def links(self, skipself=False):
        """Returns a generator of all links under the hierarchy.

        Args:
            skipself (bool): If ``True``, then the generator skips this link
                and starts with the first child link.

        Returns:
            A generator object that generates all links.

        """
        if not skipself:
            yield self

    def namedlinks(self, skipself=False):
        """Returns a generator of all (path, link) pairs under the hierarchy.

        Args:
            skipself (bool): If ``True``, then the generator skips this link
                and starts with the first child link.

        Returns:
            A generator object that generates all (path, link) pairs.

        """
        if not skipself:
            yield '/', self

    def children(self):
        """Returns a generator of all child links.

        Returns:
            A generator object that generates all child links.

        """
        if 0:
            yield

    def copyparams(self, link):
        """Copies all parameters from given link.

        This method copies data arrays of all parameters in the hierarchy. The
        copy is even done across the host and devices. Note that this method
        does not copy the gradient arrays.

        Args:
            link (Link): Source link object.

        """
        src = link.__dict__
        dst = self.__dict__
        for name in self._params:
            dst[name].copydata(src[name])

    def cleargrads(self):
        """Clears all gradient arrays.

        This method should be called before the backward computation at every
        iteration of the optimization.

        """
        for param in self.params():
            param.cleargrad()

    def zerograds(self):
        """Initializes all gradient arrays by zero.

        This method can be used for the same purpose of cleargrads, but less
        efficient. This method is left for backward compatibility.

        .. deprecated:: v1.15
           Use :meth:`cleargrads` instead.

        """
        warnings.warn(
            'Link.zerograds is deprecated. Use Link.cleargrads instead.',
            DeprecationWarning)
        for param in self.params():
            param.zerograd()

    def addgrads(self, link):
        """Accumulates gradient values from given link.

        This method adds each gradient array of the given link to corresponding
        gradient array of this link. The accumulation is even done across
        host and different devices.

        Args:
            link (Link): Source link object.

        """
        src = link.__dict__
        dst = self.__dict__
        for name in self._params:
            dst[name].addgrad(src[name])

    def enable_update(self):
        """Enables update rules of all parameters under the link hierarchy.

        This method sets the :attr:`~chainer.UpdateRule.enabled` flag of the
        update rule of each parameter variable to ``True``.

        """
        for param in self.params():
            rule = param.update_rule
            if rule is not None:
                rule.enabled = True

    def disable_update(self):
        """Disables update rules of all parameters under the link hierarchy.

        This method sets the :attr:~chainer.UpdateRule.enabled` flag of the
        update rule of each parameter variable to ``False``.

        """
        for param in self.params():
            rule = param.update_rule
            if rule is not None:
                rule.enabled = False

    @property
    def update_enabled(self):
        """``True`` if at least one parameter has an update rule enabled."""
        for param in self.params():
            rule = param.update_rule
            if rule is not None and rule.enabled:
                return True
        return False

    def serialize(self, serializer):
        """Serializes the link object.

        Args:
            serializer (~chainer.AbstractSerializer): Serializer object.

        """
        d = self.__dict__
        for name in self._params:
            param = d[name]
            data = serializer(name, param.data)
            if param.data is None and data is not None:
                # Initialize the parameter here
                param.initialize(data.shape)
                if isinstance(param.data, numpy.ndarray):
                    numpy.copyto(param.data, data)
                else:
                    param.data.set(numpy.asarray(data))
        for name in self._persistent:
            d[name] = serializer(name, d[name])


class Chain(Link):

    """Composable link with object-like interface.

    Composability is one of the most important features of neural nets. Neural
    net models consist of many reusable fragments, and each model itself might
    be embedded into a larger learnable system. Chain enables us to write a
    neural net based on composition, without bothering about routine works like
    collecting parameters, serialization, copying the structure with parameters
    shared, etc.

    This class actually provides a way to compose one or more links into one
    structure. A chain can contain one or more *child links*. Child link is a
    link registered to the chain with its own name. The child link is stored to
    an attribute of the chain with the name. User can write a whole model or a
    fragment of neural nets as a child class of Chain.

    Each chain itself is also a link. Therefore, one can combine chains into
    higher-level chains. In this way, links and chains construct a *link
    hierarchy*. Link hierarchy forms a tree structure, where each node is
    identified by the path from the root. The path is represented by a string
    like a file path in UNIX, consisting of names of nodes on the path, joined
    by slashes ``/``.

    A child link can be added just by assigning it to an attribute of the
    chain within :meth:`an initialization scope <chainer.Link.init_scope>`.

    The registered child link is saved and loaded on serialization and
    deserialization, and involved in the optimization. The registered link
    is called a child. The child link is accessible via :meth:`children`
    generator, which returns a generator running through the children in
    registered order.

    On registration of a child link, its :attr:`~Link.name` attribute is also
    set (or overwritten if the link has already been registered to another
    chain).

    .. admonition:: Example

       This is a simple example of custom chain definition. Chainer itself also
       provides some chains defined under the :mod:`~chainer.links` module.
       They might serve as examples, too.

       Consider we want to define a multi-layer perceptron consisting of two
       hidden layers with rectifiers as activation functions. We can use the
       :class:`~chainer.links.Linear` link as a building block::

          import chainer
          import chainer.functions as F
          import chainer.links as L

          class MultiLayerPerceptron(chainer.Chain):

              def __init__(self, n_in, n_hidden, n_out):
                  super(MultilayerPerceptron, self).__init__()
                  with self.init_scope():
                      self.layer1 = L.Linear(n_in, n_hidden)
                      self.layer2 = L.Linear(n_hidden, n_hidden)
                      self.layer3 = L.Linear(n_hidden, n_out)

              def __call__(self, x):
                  # Forward propagation
                  h1 = F.relu(self.layer1(x))
                  h2 = F.relu(self.layer2(h1))
                  return self.layer3(h2)

       Child links are registered via the assignment within a
       ``with self.init_scope():`` block. The forward propagation is often
       implemented as The ``__call__`` operator as the above example, though
       it is not mandatory.

    Args:
        links: Child links. The keywords are used as their names. The names are
            also set to the links.

            .. deprecated:: v2.0.0

               Assign child links directly to attributes, instead.

    """

    def __init__(self, **links):
        super(Chain, self).__init__()
        self._children = set()

        for name, link in six.iteritems(links):
            self.add_link(name, link)

    def __getitem__(self, name):
        """Equivalent to getattr."""
        return getattr(self, name)

    def __setattr__(self, name, value):
        if self.within_init_scope and isinstance(value, Link):
            if hasattr(self, name):
                raise AttributeError(
                    'cannot register a new link %s: attribute exists' % name)
            value.name = name
            self._children.add(name)
        super(Chain, self).__setattr__(name, value)

    def __delattr__(self, name):
        self._children.discard(name)
        super(Chain, self).__delattr__(name)

    def add_link(self, name, link):
        """Registers a child link to this chain.

        .. deprecated:: v2.0.0

           Assign the child link directly to an attribute within
           :meth:`an initialization scope <chainer.Link.init_scope>`, instead.
           For example, the following code

           .. code-block:: python

              chain.add_link('l1', L.Linear(3, 5))

           can be replaced by the following line.

           .. code-block:: python

              with self.init_scope():
                  chain.l1 = L.Linear(3, 5)

           The latter one is easier for IDEs to keep track of the attribute's
           type.

        Args:
            name (str): Name of the child link. This name is also used as the
                attribute name.
            link (Link): The link object to be registered.

        """
        warnings.warn('''\
Child link registeration via Chain.__init__ and Chain.add_link are deprecated.
Assign a Link object directly to an attribute within a \
"with link.init_scope():" block instead.
        ''', DeprecationWarning)
        if name in self.__dict__:
            raise AttributeError(
                'cannot register a new link %s: attribute exists' % name)
        if not isinstance(link, Link):
            raise TypeError('cannot register a non-link object as a child')
        with self.init_scope():
            setattr(self, name, link)

    def copy(self):
        ret = super(Chain, self).copy()
        ret._children = set(ret._children)
        d = ret.__dict__
        for name in ret._children:
            # copy child links recursively
            copied = d[name].copy()
            copied.name = name
            d[name] = copied
        return ret

    def to_cpu(self):
        super(Chain, self).to_cpu()
        d = self.__dict__
        for name in self._children:
            d[name].to_cpu()
        return self

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            super(Chain, self).to_gpu()
            d = self.__dict__
            for name in self._children:
                d[name].to_gpu()
        return self

    def params(self, include_uninit=True):
        for param in super(Chain, self).params(include_uninit):
            yield param
        d = self.__dict__
        for name in self._children:
            for param in d[name].params(include_uninit):
                yield param

    def namedparams(self, include_uninit=True):
        for ret in super(Chain, self).namedparams(include_uninit):
            yield ret
        d = self.__dict__
        for name in self._children:
            prefix = '/' + name
            for path, param in d[name].namedparams(include_uninit):
                yield prefix + path, param

    def links(self, skipself=False):
        if not skipself:
            yield self
        d = self.__dict__
        for name in self._children:
            for link in d[name].links():
                yield link

    def namedlinks(self, skipself=False):
        if not skipself:
            yield '/', self
        d = self.__dict__
        for name in self._children:
            child = d[name]
            prefix = '/' + name
            yield prefix, child
            for path, link in d[name].namedlinks(True):
                yield prefix + path, link

    def children(self):
        d = self.__dict__
        for name in self._children:
            yield d[name]

    def copyparams(self, link):
        super(Chain, self).copyparams(link)
        src = link.__dict__
        dst = self.__dict__
        for name in self._children:
            dst[name].copyparams(src[name])

    def addgrads(self, link):
        super(Chain, self).addgrads(link)
        src = link.__dict__
        dst = self.__dict__
        for name in self._children:
            dst[name].addgrads(src[name])

    def serialize(self, serializer):
        super(Chain, self).serialize(serializer)
        d = self.__dict__
        for name in self._children:
            d[name].serialize(serializer[name])


class ChainList(Link):

    """Composable link with list-like interface.

    This is another example of compositional link. Unlike :class:`Chain`, this
    class can be used like a list of child links. Each child link is indexed by
    a non-negative integer, and it maintains the current number of registered
    child links. The :meth:`add_link` method inserts a new link at the end of
    the list. It is useful to write a chain with arbitrary number of child
    links, e.g. an arbitrarily deep multi-layer perceptron.

    Note that this class does not implement all methods of :class:`list`.

    Args:
        links: Initial child links.

    """

    def __init__(self, *links):
        super(ChainList, self).__init__()
        self._children = []

        for link in links:
            self.add_link(link)

    def __getitem__(self, index):
        """Returns the child at given index.

        Args:
            index (int): Index of the child in the list.

        Returns:
            Link: The ``index``-th child link.

        """
        return self._children[index]

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        """Returns the number of children."""
        return len(self._children)

    def append(self, link):
        """Registers a child link and adds it to the tail of the list.

        This is equivalent to :meth:`add_link`. This method has been added to
        emulate the ``list`` interface.

        Args:
            link (Link): The link object to be regsitered.

        """
        self.add_link(link)

    def add_link(self, link):
        """Registers a child link and adds it to the tail of the list.

        Args:
            link (Link): The link object to be registered.

        """
        link.name = str(len(self._children))
        self._children.append(link)

    def copy(self):
        ret = super(ChainList, self).copy()
        ret._children = list(ret._children)  # copy
        children = ret._children
        for i, child in enumerate(children):
            child = child.copy()
            child.name = str(i)
            children[i] = child
        return ret

    def to_cpu(self):
        super(ChainList, self).to_cpu()
        for link in self._children:
            link.to_cpu()
        return self

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            super(ChainList, self).to_gpu()
            for link in self._children:
                link.to_gpu()
        return self

    def params(self, include_uninit=True):
        for param in super(ChainList, self).params(include_uninit):
            yield param
        for link in self._children:
            for param in link.params(include_uninit):
                yield param

    def namedparams(self, include_uninit=True):
        for ret in super(ChainList, self).namedparams(include_uninit):
            yield ret
        for idx, link in enumerate(self._children):
            prefix = '/%d' % idx
            for path, param in link.namedparams(include_uninit):
                yield prefix + path, param

    def links(self, skipself=False):
        if not skipself:
            yield self
        for child in self._children:
            for link in child.links():
                yield link

    def namedlinks(self, skipself=False):
        if not skipself:
            yield '/', self
        for idx, child in enumerate(self._children):
            prefix = '/%d' % idx
            yield prefix, child
            for path, link in child.namedlinks(True):
                yield prefix + path, link

    def children(self):
        for child in self._children:
            yield child

    def copyparams(self, link):
        super(ChainList, self).copyparams(link)
        for idx, child in enumerate(self._children):
            child.copyparams(link[idx])

    def addgrads(self, link):
        super(ChainList, self).addgrads(link)
        for idx, child in enumerate(self._children):
            child.addgrads(link[idx])

    def serialize(self, serializer):
        super(ChainList, self).serialize(serializer)
        for idx, child in enumerate(self._children):
            child.serialize(serializer['%d' % idx])
