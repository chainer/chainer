import collections
import contextlib
import copy
import warnings

import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import initializers
from chainer import link_hook
from chainer.utils import collections_abc
from chainer import variable
import chainerx


def _is_shape(value):
    if value is None:
        return True
    elif isinstance(value, collections_abc.Sequence):
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
        return value, numpy.float32
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
                          initializers.Normal(), (n_out, n_in))
                      self.b = chainer.Parameter(
                          initializers.Zero(), (n_out,))

              def forward(self, x):
                  return F.linear(x, self.W, self.b)

       This example shows that a user can define arbitrary parameters and use
       them in any methods. Links typically implement the ``forward``
       operator, although they can also provide other methods to implement the
       forward propagation.

    Args:
        params:
            Names, shapes, and optional dtypes of initial parameters.
            The keywords are used as the parameter names and the corresponding
            values consist either of the shape or a tuple of shape and a dtype
            ``(shape, dtype)``.
            If only the shape is supplied, the default dtype will be used.

    Attributes:
        name (str): Name of this link, given by the parent chain (if exists).

    """

    _local_link_hooks = None

    def __init__(self, **params):
        self._params = set()
        self._persistent = set()
        self._device = backend.CpuDevice()
        self._within_init_scope = False
        self.name = None

        for name, value in six.iteritems(params):
            shape, dtype = _ensure_shape_dtype(value)
            self.add_param(name, shape, dtype=dtype)

    @property
    def local_link_hooks(self):
        """Ordered dictionary of registered link hooks.

        Contrary to ``chainer.thread_local.link_hooks``,
        which registers its elements to all functions,
        link hooks in this property are specific to this link.

        """
        if self._local_link_hooks is None:
            self._local_link_hooks = collections.OrderedDict()
        return self._local_link_hooks

    @property
    def _n_local_link_hooks(self):
        return (0 if self._local_link_hooks is None
                else len(self._local_link_hooks))

    @property
    def device(self):
        return self._device

    @property
    def _device_id(self):
        warnings.warn(
            'Link._device_id is left only for backward compatibility and '
            'likely to be removed. Use Link.device instead.',
            DeprecationWarning)
        if self._device.xp is cuda.cupy:
            return self._device.device.id
        return None

    @property
    def xp(self):
        """Array module for this link.

        Depending on which of CPU/GPU this link is on, this property returns
        :mod:`numpy` or :mod:`cupy`.

        """
        return self._device.xp

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

    def __call__(self, *args, **kwargs):

        # TODO(niboshi): Support link hooks for other forward methods.
        hooks = chainer._get_link_hooks()
        if self._n_local_link_hooks > 0:
            hooks = collections.OrderedDict(hooks)
            hooks.update(self.local_link_hooks)
        hooks = hooks.values()  # avoid six for performance

        # Call forward_preprocess hook
        if hooks:
            cb_args = link_hook._ForwardPreprocessCallbackArgs(
                self, 'forward', args, kwargs)
            for hook in hooks:
                hook.forward_preprocess(cb_args)

        # Call the forward function
        # (See #5078) super().__call__ is used when the method is injected by a
        # mixin class. To keep backward compatibility, the injected one is
        # prioritized over forward().
        forward = getattr(super(Link, self), '__call__', None)
        if forward is None:
            forward = self.forward
        out = forward(*args, **kwargs)

        # Call forward_postprocess hook
        if hooks:
            cb_args = link_hook._ForwardPostprocessCallbackArgs(
                self, 'forward', args, kwargs, out)
            for hook in hooks:
                hook.forward_postprocess(cb_args)

        return out

    def __setattr__(self, name, value):
        if self.within_init_scope and isinstance(value, variable.Parameter):
            value.name = name
            value.to_device(self._device)
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

    def copy(self, mode='share'):
        """Copies the link hierarchy to new one.

        The whole hierarchy rooted by this link is copied. There are three
        modes to perform copy. Please see the documentation for the argument
        ``mode`` below.

        The name of the link is reset on the copy, since the copied instance
        does not belong to the original parent chain (even if exists).

        Args:
            mode (str): It should be either ``init``, ``copy``, or ``share``.
                ``init`` means parameter variables under the returned link
                object is re-initialized by calling their
                :meth:`~chainer.Parameter.initialize` method, so that all the
                parameters may have different initial values from the original
                link.
                ``copy`` means that the link object is deeply copied, so that
                its parameters are not re-initialized but are also deeply
                copied. Thus, all parameters have same initial values but can
                be changed independently.
                ``share`` means that the link is shallowly copied, so that its
                parameters' arrays are shared with the original one. Thus,
                their values are changed synchronously. The default ``mode``
                is ``share``.

        Returns:
            Link: Copied link object.

        """
        if mode == 'share':
            ret = copy.copy(self)
            ret._params = set(self._params)
            ret._persistent = set(self._persistent)
            ret.name = None
            d = ret.__dict__
            for name in ret._params:
                d[name] = copy.copy(d[name])
                d[name].grad = None
            return ret
        elif mode == 'copy':
            return copy.deepcopy(self)
        elif mode == 'init':
            ret = copy.deepcopy(self)
            for param in ret.params(include_uninit=False):
                param.initialize(param.shape)
            return ret
        else:
            raise ValueError(
                'The \'mode\' argument should be either \'init\','
                '\'copy\', or \'share\'. But {} was given.'.format(mode))

    def to_cpu(self):
        """Copies parameter variables and persistent values to CPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to CPU, the link implementation must
        override :meth:`Link.to_device` to do so.

        Returns: self

        """
        return self.to_device(backend.CpuDevice())

    def to_gpu(self, device=None):
        """Copies parameter variables and persistent values to GPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to GPU, the link implementation must
        override :meth:`Link.to_device` to do so.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        Returns: self

        """
        cuda.check_cuda_available()
        return self._to_device(
            cuda._get_device_or_current(device),
            skip_between_cupy_devices=True)

    def to_intel64(self):
        """Copies parameter variables and persistent values to CPU."""
        intel64.check_ideep_available()
        return self.to_device(intel64)

    def to_chainerx(self):
        """Converts parameter variables and persistent values to ChainerX \
without any copy.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to ChainerX, the link implementation must
        override this method to do so.

        Returns: self
        """  # NOQA
        if not chainerx.is_available():
            raise RuntimeError('ChainerX is not available.')

        xp = self._device.xp
        if xp is chainerx:
            return self

        d = self.__dict__
        for name in self._params:
            d[name].to_chainerx()
        for name in self._persistent:
            if not numpy.isscalar(d[name]):
                d[name] = backend.to_chainerx(d[name])

        self._device = (
            backend.ChainerxDevice.from_fallback_device(self._device))

        return self

    def from_chainerx(self):
        """Converts parameter variables and persistent values from ChainerX \
to NumPy/CuPy devices without any copy."""
        d = self.__dict__
        for name in self._params:
            d[name].from_chainerx()
        for name in self._persistent:
            if not numpy.isscalar(d[name]):
                d[name] = backend.from_chainerx(d[name])

        if isinstance(self._device, backend.ChainerxDevice):
            self._device = self._device.fallback_device

        return self

    def to_device(self, device):
        """Copies parameter variables and persistent values to the specified \
device.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to the device, the link implementation must
        override this method to do so.

        Args:
            device: Target device specifier. See
                :func:`~chainer.get_device` for available values.

        Returns: self

        """  # NOQA
        return self._to_device(device, skip_between_cupy_devices=False)

    def _to_device(self, device, skip_between_cupy_devices=False):
        # `skip_between_cupy_devices` argument is a workaround
        # for `Link.to_gpu` which does not transfer cupy parameters to
        # a different CUDA device.
        device = chainer.get_device(device)

        d = self.__dict__
        for name in self._params:
            if not (skip_between_cupy_devices
                    and device.xp is cuda.cupy
                    and d[name].device.xp is cuda.cupy):
                d[name].to_device(device)
        for name in self._persistent:
            if not numpy.isscalar(d[name]):
                if not (skip_between_cupy_devices
                        and device.xp is cuda.cupy
                        and isinstance(d[name], cuda.ndarray)):
                    d[name] = device.send(d[name])

        self._device = device
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
        for name in sorted(self._params):
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
        for name in sorted(self._params):
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

    def copyparams(self, link, copy_persistent=True):
        """Copies all parameters from given link.

        This method copies data arrays of all parameters in the hierarchy. The
        copy is even done across the host and devices. Note that this method
        does not copy the gradient arrays.

        *From v5.0.0:* this method also copies the persistent values (e.g. the
        moving statistics of :class:`~chainer.links.BatchNormalization`). If
        the persistent value is an ndarray, the elements are copied. Otherwise,
        it is copied using :func:`copy.deepcopy`. The old behavior (not copying
        persistent values) can be reproduced with ``copy_persistent=False``.

        Args:
            link (Link): Source link object.
            copy_persistent (bool): If ``True``, persistent values are also
                copied. ``True`` by default.

        """
        src = link.__dict__
        dst = self.__dict__
        for name in self._params:
            dst[name].copydata(src[name])
        if copy_persistent:
            array_types = chainer.get_array_types()
            for name in self._persistent:
                d = dst[name]
                s = src[name]
                if isinstance(d, array_types) and isinstance(s, array_types):
                    backend.copyto(d, s)
                else:
                    dst[name] = copy.deepcopy(s)

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

        This method sets the :attr:`~chainer.UpdateRule.enabled` flag of the
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

    def repeat(self, n_repeat, mode='init'):
        """Repeats this link multiple times to make a :class:`~chainer.Sequential`.

        This method returns a :class:`~chainer.Sequential` object which has
        the same :class:`~chainer.Link` multiple times repeatedly. The ``mode``
        argument means how to copy this link to repeat.

        .. admonition:: Example

            You can repeat the same link multiple times to create a longer
            :class:`~chainer.Sequential` block like this:

            .. testcode::

                class ConvBNReLU(chainer.Chain):

                    def __init__(self):
                        super(ConvBNReLU, self).__init__()
                        with self.init_scope():
                            self.conv = L.Convolution2D(
                                None, 64, 3, 1, 1, nobias=True)
                            self.bn = L.BatchNormalization(64)

                    def forward(self, x):
                        return F.relu(self.bn(self.conv(x)))

                net = ConvBNReLU().repeat(16, mode='init')

            The ``net`` object contains 16 blocks, each of which is
            ``ConvBNReLU``. And the ``mode`` was ``init``, so each block
            is re-initialized with different parameters. If you give
            ``copy`` to this argument, each block has same values for its
            parameters but its object ID is different from others. If it is
            ``share``, each block is same to others in terms of not only
            parameters but also the object IDs because they are shallow-copied,
            so that when the parameter of one block is changed, all the
            parameters in the others also change.

        Args:
            n_repeat (int): Number of times to repeat.
            mode (str): It should be either ``init``, ``copy``, or ``share``.
                ``init`` means parameters of each repeated element in the
                returned :class:`~chainer.Sequential` will be re-initialized,
                so that all elements have different initial parameters.
                ``copy`` means that the parameters will not be re-initialized
                but object itself will be deep-copied, so that all elements
                have same initial parameters but can be changed independently.
                ``share`` means all the elements which consist the resulting
                :class:`~chainer.Sequential` object are same object because
                they are shallow-copied, so that all parameters of elements
                are shared with each other.

        """
        ret = chainer.Sequential()
        if n_repeat <= 0:
            return ret
        if mode not in ['init', 'copy', 'share']:
            raise ValueError(
                'The \'mode\' argument should be either \'init\','
                '\'copy\', or \'share\'. But {} was given.'.format(mode))
        link = self
        for _ in range(n_repeat):
            ret.append(link.copy(mode))
        return ret

    def count_params(self):
        """Counts the total number of parameters.

        This method counts the total number of scalar values included in all
        the :class:`~chainer.Parameter`\\ s held by this link and its
        descendants.

        If the link containts uninitialized parameters, this method raises a
        warning.

        Returns:
            The total size of parameters (int)

        """

        size = 0
        for name, param in self.namedparams():
            if param.array is None:
                warnings.warn(
                    'Parameter \'{}\' has not been initialized, so the '
                    'resulting count will not include the number of parameters'
                    ' in it.'.format(name))
                continue
            size += param.size
        return size

    def add_hook(self, hook, name=None):
        """Registers a link hook.

        Args:
            hook (~chainer.LinkHook): Link hook to be registered.
            name (str): Name of the link hook. The name must be unique
                among link hooks registered to this link. If ``None``,
                the default name of the link hook is used.

        Returns:
            self

        """
        if not isinstance(hook, link_hook.LinkHook):
            raise TypeError('Hook must be of type LinkHook')
        if name is None:
            name = hook.name
        hooks = self.local_link_hooks
        if name in hooks:
            raise KeyError('Hook %s already exists' % name)
        hooks[name] = hook
        hook.added(self)
        return self

    def delete_hook(self, name):
        """Unregisters the link hook.

        Args:
            name (str): The name of the link hook to be unregistered.

        """
        if name in self.local_link_hooks:
            self.local_link_hooks[name].deleted(self)
            del self.local_link_hooks[name]
        else:
            raise KeyError('Hook %s does not exist' % name)


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
    chain within :meth:`~chainer.Chain.init_scope`.

    The registered child link is saved and loaded on serialization and
    deserialization, and involved in the optimization. The registered link
    is called a child. The child link is accessible via :meth:`children`
    generator, which returns a generator running through the children in
    lexical order.

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
                  super(MultiLayerPerceptron, self).__init__()
                  with self.init_scope():
                      self.layer1 = L.Linear(n_in, n_hidden)
                      self.layer2 = L.Linear(n_hidden, n_hidden)
                      self.layer3 = L.Linear(n_hidden, n_out)

              def forward(self, x):
                  # Forward propagation
                  h1 = F.relu(self.layer1(x))
                  h2 = F.relu(self.layer2(h1))
                  return self.layer3(h2)

       Child links are registered via the assignment within a
       ``with self.init_scope():`` block. The forward propagation is often
       implemented as the ``forward`` operator as the above example, though
       it is not mandatory.

    Args:
        links: Child links. The keywords are used as their names. The names are
            also set to the links.

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

        Args:
            name (str): Name of the child link. This name is also used as the
                attribute name.
            link (Link): The link object to be registered.

        """
        if name in self.__dict__:
            raise AttributeError(
                'cannot register a new link %s: attribute exists' % name)
        if not isinstance(link, Link):
            raise TypeError('cannot register a non-link object as a child')
        with self.init_scope():
            setattr(self, name, link)

    def copy(self, mode='share'):
        ret = super(Chain, self).copy()
        ret._children = set(ret._children)
        d = ret.__dict__
        for name in ret._children:
            # copy child links recursively
            copied = d[name].copy(mode)
            copied.name = name
            d[name] = copied
        return ret

    def to_chainerx(self):
        super(Chain, self).to_chainerx()
        d = self.__dict__
        for name in self._children:
            d[name].to_chainerx()
        return self

    def from_chainerx(self):
        super(Chain, self).from_chainerx()
        d = self.__dict__
        for name in self._children:
            d[name].from_chainerx()
        return self

    def _to_device(self, device, skip_between_cupy_devices=False):
        # Overrides Link._to_device

        device = chainer.get_device(device)
        super(Chain, self)._to_device(
            device, skip_between_cupy_devices=skip_between_cupy_devices)
        d = self.__dict__
        for name in self._children:
            d[name]._to_device(
                device, skip_between_cupy_devices=skip_between_cupy_devices)
        return self

    def params(self, include_uninit=True):
        for param in super(Chain, self).params(include_uninit):
            yield param
        d = self.__dict__
        for name in sorted(self._children):
            for param in d[name].params(include_uninit):
                yield param

    def namedparams(self, include_uninit=True):
        for ret in super(Chain, self).namedparams(include_uninit):
            yield ret
        d = self.__dict__
        for name in sorted(self._children):
            prefix = '/' + name
            for path, param in d[name].namedparams(include_uninit):
                yield prefix + path, param

    def links(self, skipself=False):
        if not skipself:
            yield self
        d = self.__dict__
        for name in sorted(self._children):
            for link in d[name].links():
                yield link

    def namedlinks(self, skipself=False):
        if not skipself:
            yield '/', self
        d = self.__dict__
        for name in sorted(self._children):
            child = d[name]
            prefix = '/' + name
            yield prefix, child
            for path, link in d[name].namedlinks(True):
                yield prefix + path, link

    def children(self):
        d = self.__dict__
        for name in sorted(self._children):
            yield d[name]

    def copyparams(self, link, copy_persistent=True):
        super(Chain, self).copyparams(link, copy_persistent)
        src = link.__dict__
        dst = self.__dict__
        for name in self._children:
            dst[name].copyparams(src[name], copy_persistent)

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


class ChainList(Link, collections_abc.MutableSequence):

    """Composable link with list-like interface.

    This is another example of compositional link. Unlike :class:`Chain`, this
    class can be used like a list of child links. Each child link is indexed by
    a non-negative integer, and it maintains the current number of registered
    child links. The :meth:`add_link` method inserts a new link at the end of
    the list. It is useful to write a chain with arbitrary number of child
    links, e.g. an arbitrarily deep multi-layer perceptron.

    This class inherits the methods `index`, `count`, `append`, `reverse`,
    `extend`, `pop`, `remove` from `collections.abc.MutableSequence` and
    can be accessed and assigned by index or slice.

    Args:
        links: Initial child links.

    """

    def __init__(self, *links):
        super(ChainList, self).__init__()
        self._children = []

        for link in links:
            self.add_link(link)

    def __setattr__(self, name, value):
        if self.within_init_scope and isinstance(value, Link):
            raise TypeError(
                'cannot register a new link'
                ' within a "with chainlist.init_scope():" block.')
        super(ChainList, self).__setattr__(name, value)

    def __setitem__(self, index, value):
        if isinstance(index, int):
            value.name = str(index)
            self._children[index] = value
        elif isinstance(index, slice):
            self._children[index] = value
            for i, c in enumerate(self._children):
                c.name = str(i)
        else:
            raise TypeError(
                'ChainList indices must be integers or slices, not %s' %
                type(index).__name__)

    def __getitem__(self, index):
        """Returns the child at given index.

        Args:
            index (int): Index of the child in the list.

        Returns:
            Link: The ``index``-th child link.

        """
        return self._children[index]

    def __delitem__(self, index):
        del self._children[index]
        for i, c in enumerate(self._children):
            c.name = str(i)

    def insert(self, index, link):
        """Insert a child link at the given index.

        Args:
            index (int): The position of the list where the new
            link is inserted.
            link (Link): The link to be inserted.

        """
        if index == len(self._children):
            self._children.append(link)
            link.name = str(index)
        else:
            self._children.insert(index, link)
            for i, c in enumerate(self._children):
                c.name = str(i)

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        """Returns the number of children."""
        return len(self._children)

    def add_link(self, link):
        """Registers a child link and adds it to the tail of the list.

        Args:
            link (Link): The link object to be registered.

        """
        self.append(link)

    def copy(self, mode='share'):
        """Returns a deep copy of the chainlist."""
        ret = super(ChainList, self).copy()
        ret._children = list(ret._children)  # copy
        children = ret._children
        for i, child in enumerate(children):
            child = child.copy(mode)
            child.name = str(i)
            children[i] = child
        return ret

    def to_chainerx(self):
        super(ChainList, self).to_chainerx()
        for link in self._children:
            link.to_chainerx()
        return self

    def _to_device(self, device, skip_between_cupy_devices=False):
        # Overrides Link._to_device

        device = chainer.get_device(device)
        super(ChainList, self)._to_device(
            device, skip_between_cupy_devices=skip_between_cupy_devices)
        for link in self._children:
            link._to_device(
                device, skip_between_cupy_devices=skip_between_cupy_devices)
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

    def copyparams(self, link, copy_persistent=True):
        super(ChainList, self).copyparams(link, copy_persistent)
        for idx, child in enumerate(self._children):
            child.copyparams(link[idx], copy_persistent)

    def addgrads(self, link):
        super(ChainList, self).addgrads(link)
        for idx, child in enumerate(self._children):
            child.addgrads(link[idx])

    def serialize(self, serializer):
        super(ChainList, self).serialize(serializer)
        for idx, child in enumerate(self._children):
            child.serialize(serializer['%d' % idx])
