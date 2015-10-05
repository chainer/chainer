import copy as copy_module
import sys

import numpy
import six

from chainer import cuda


class Link(object):

    """Building block of neural network models.

    Link is a building block of neural network models that support various
    features like handling parameters, defining network fragments,
    serialization, etc. Typical links define some network fragments with
    parameters (e.g. fully-connected layer with internal parameters,
    multi-layer perceptron, stateful RNN layer, autoencoder with given encoder
    and decoder, etc.). Links can be used to construct larger networks (like
    creating a chain from small links).

    There are two types of links: *premitive links* and *compositional links*.
    Primitive links are links that do not use other links inside of them. Such
    links typically inherit :class:`Link` directly, and define network
    fragments with parameters and states. Compositional links are links that
    use other links inside of them. Such links typically inherit
    :class:`DictLink` or :class:`ListLink`, which provide ways to construct
    *link hierarchy* as a tree structure.

    Typical neural network models are defined by the link hierarchies. For
    example, a deep autoencoder is defined as a link that holds an encoder
    and a decoder as internal links, each of which is a deep neural network
    that consists of many layers. A layer with parameters is defined as a link,
    so the encoder and the decoder hold many links.

    In a link hierarchy, each link is given a unique name reflecting the path
    from the *root link*. For example, if the autoencoder contains an encoder
    by the relative name ``"encoder"``, and the autoencoder is the root link,
    then the name of the encoder is ``"/encoder"``. If the encoder names the
    first linear layer ``"l1"``, then its name in the full hirarchy is
    ``"/encoder/l1"``. The root link always has the name ``"/"``.

    Each link can contain parameters and states. Each parameter is a
    :class:`Variable` object. Parameters are used as arguments of some
    :class:`Function` implementation. Each state is an array, and can be used
    freely in the link definition, but it should be an array whose value is
    updated by the learning process, i.e. it should be *a part of learning
    result like parameters*.

    Parameters and states also have their own names. The relative names are
    used to construct absolute names in the link hierarchy. For example, in the
    above autoencoder case, if the layer ``"l1"`` contains a parameter named
    ``"W"``, then its absolute name is ``"/encoder/l1/_params/W"``. The
    absolute names of states are also constructed in this way (while
    ``"_params"`` is replaced by ``"_states"``).

    Link supports the serialization protocol of Chainer. **Note that only
    parameters and states are saved and loaded**. In Chainer, other attributes
    are considered as a part of user program (i.e. a part of network
    definition). In order to construct a link from saved file, other attributes
    must be identically reconstructed by user codes.

    .. admonition:: Example

       Consider that we want to define a simple primitive link that implements
       a fully-connected layer based on the :func:`~functions.linear` function.
       Note that this function takes input units, a weight variable, and a bias
       variable as arguments. Then, the fully-connected layer can be defined as
       follows::

          import chainer
          import chainer.functions as F
          import numpy as np

          class LinearLayer(chainer.Link):
              def __init__(self, n_in, n_out):
                  super(LinearLayer, self).__init__()
                  self.params['W'] = chainer.Variable(
                      np.random.randn(n_out, n_in).astype('f'))
                  self.params['b'] = chainer.Variable(
                      np.zeros(n_out, dtype='f'))

              def __call__(self, x):
                  return F.linear(x, self.params['W'], self.params['b'])

       It shows that a user can define arbitrary parameters and use them in any
       methods. Primitive links typically implement the ``__call__`` operator.

       Note that more useful version of the linear link is provided by the
       :class:`~functions.Linear` class.

    Args:
        name (str): The absolute name of this link in the hierarchy.
            The name can also be updated by the setter of the :attr:`name`
            property.

    Attributes:
        params (dict): Parameter variables with names.
        states (dict): State arrays with names.

    """
    def __init__(self, name=''):
        self.params = {}
        self.states = {}
        self._name = name

    @property
    def name(self):
        """Absolute name of this link in the hierarchy.

        The name reflects the link hierarchy by a slash (``/``) separated path
        like a UNIX file system. The root link has the name ``"/"``. A non-root
        link is held by another link with some relative name, where its
        absolute name is a concatenation of that of *parent* link and the
        relative name separated by a slash.

        The setter just replaces the name of this link.

        .. note::
           :class:`DictLink` and :class:`ListLink` override the setter, where
           the names of all *child* links are also updated recursively.

        """
        return self._name or '/'

    @name.setter
    def name(self, name):
        self._name = name if name != '/' else ''

    @property
    def volatile(self):
        """Volatility flag of internal parameters.

        It is True if at least one of the parameters under this link is
        volatile. Otherwise, it is False.

        This property also has a setter, which updates the volatile flag of
        all parameters under this link.

        """
        for _, param in self.visitparams():
            return param.volatile
        return False

    @volatile.setter
    def volatile(self, value):
        value = bool(value)
        for _, param in self.visitparams():
            param.volatile = value

    def copy(self, shared=True):
        """Copies the link hierarchy starting from this link.

        Note that the returned link becomes root.

        .. note::
           Actually, Link.copy just copies the link itself. Subclasses that
           builds a link hierarchy (e.g. :class:`DictLink` and
           :class:`ListLink`) override this method to copy the whole hierarchy.

        Args:
            shared (bool): If True, parameters and states are shared.
                Otherwise, they are deeply copied.

        Returns:
            Link: A copy of the link object.

        """
        ret = self._copy_params_states(shared)
        ret.name = ''
        return ret

    def _copy_params_states(self, shared):
        ret = copy_module.copy(self)
        copy = copy_module.copy if shared else copy_module.deepcopy
        ret.params = {}
        for key, param in six.iteritems(self.params):
            ret.params[key] = copy(param)
        ret.states = copy(self.states)
        return ret

    def to_cpu(self):
        """Copies all parameters and states to CPU.

        This method copies all parameters and states in the link hierarchy to
        CPU.

        .. note::
           Use this method with ``copy(shared=True)`` to create a CPU version
           of the link without affecting its GPU version.

        Returns: self

        """
        for link in self.visitlinks():
            for param in six.itervalues(link.params):
                param.data = cuda.to_cpu(param.data)
                param._grad = cuda.to_cpu(param._grad)

            states = link.states
            for key, value in six.iteritems(states):
                states[key] = cuda.to_cpu(value)

        return self

    def to_gpu(self, device=None):
        """Copies all parameters and states to GPU if needed.

        This method copies all parameters and states in the link hierarchy to
        specified GPU device.

        .. note::
           Use this method with ``copy(shared=True)`` to create a GPU version
           of the link without affecting its CPU version.

        Args:
            device: Device specifier.

        Returns: self

        """
        cupy = cuda.cupy
        with cuda.get_device(device):
            for link in self.visitlinks():
                for param in six.itervalues(link.params):
                    param.data = cupy.asarray(param.data)
                    if param._grad is not None:
                        param._grad = cupy.asarray(param._grad)

                states = link.states
                for key, value in six.iteritems(states):
                    states[key] = cupy.asarray(value)

        return self

    def visitparams(self):
        """Generates all parameters in the link hierarchy.

        This is a generator method that iterates over all parameters in the
        link hierarchy starting from this link. It generates tuples
        ``(key, param)``, where ``param`` is a parameter variable and ``key``
        is its absolute name in the link hierarchy.

        """
        for link in self.visitlinks():
            prefix = link._name + '/_params/'
            for key, param in six.iteritems(link.params):
                yield prefix + key, param

    def visitlinks(self):
        """Generates all links under the link hierarchy.

        This is a generator method that traverse the link hierarchy and
        generates all visited links.

        .. note::
           Actually, Link.visitlinks just generates itself. Subclasses of
           Link that builds a link hierarchy (e.g. :class:`DictLink` and
           :class:`ListLink`) override this method to traverse the hierarchy.

        """
        yield self

    def copyparams(self, link):
        """Copies parameters from another link with same link hierarchy.

        This is useful to share the parameter values between multiple links.
        This procedure is required, for example, in data-parallel model
        learning. This method allows copy over multiple GPU devices.

        Args:
            link (Link): Source link object. It must have parameters of same
                names as those of the target link.

        """
        params = {}
        for path, param in link.visitparams():
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
        """Initializes the gradients of all parameters by zeros.

        It initializes the :attr:`~Variable.grad` attribute of each parameter
        by zero array if not exists. Otherwise, it just fills the gradient
        array by zeros.

        This method should be called before the backward computation.

        """
        for link in self.visitlinks():
            params = link.params
            for key, p in six.iteritems(params):
                arr = p._grad
                if arr is None:
                    data = p.data
                    xp = cuda.get_array_module(data)
                    p._grad = xp.zeros_like(data)
                else:
                    arr.fill(0)

    def addgrads(self, link):
        """Accumulates gradients from another link.

        It accumulates the gradient values of ``link`` to those of ``self``.
        This method is used to implement data-parallel model learning on GPUs.
        This method allows accumulation over multiple GPU devices.

        Args:
            link (Link): Source link object. It must have parameters of same
                names as those of the target link.

        """
        grads = {}
        for path, param in link.visitparams():
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
        """Serializes parameters and states of this link by a given serializer.

        Note that this method does not (de)serialize gradients and attributes
        other than parameters and states.

        Args:
            serializer (Serializer): Serializer object.

        """
        p = serializer['_params']
        for key, param in six.iteritems(self.params):
            param.data = p(key, param.data)
            # grad is not serialized

        states = self.states
        s = serializer['_states']
        for key, state in six.iteritems(states):
            states[key] = s(key, state)


class DictLink(Link):

    """Dictionary-like compositional link.

    DictLink is used to build a link hierarchy by string names. It has an
    interface similar to the Python :class:`dict`. The values of the "dict" are
    restricted to links, which are considered as *child* links.

    Args:
        kwds (dict): Initial set of links with relative names.
    
    """
    def __init__(self, **kwds):
        Link.__init__(self)
        self.children = kwds

        prefix = self._name + '/'
        for key, link in six.iteritems(kwds):
            if not isinstance(link, Link):
                raise TypeError('Cannot set a non-link object to DictLink')
            if link._name:
                raise ValueError('Cannot set a link to multiple parents')
            link.name = prefix + key

    def __contains__(self, key):
        """Returns True if given key exists in the link dictionary."""
        return key in self.children

    def __delitem__(self, key):
        """Deletes the link of given key from the hierarchy.

        It also sets the name of the deleted link to root (``"/"``).

        """
        self.children[key].name = ''
        del self.children[key]

    def __iter__(self):
        """Returns an iterator over the link dictionary."""
        return self.children.__iter__()

    def __getitem__(self, key):
        """Returns the link of given key."""
        return self.children[key]

    def __len__(self):
        """Returns the number of *child* links."""
        return len(self.children)

    def __setitem__(self, key, value):
        """Sets a link with given relative name.

        It also updates the name of the given link to an appropriate path in
        the link hierarchy. If the key is already used, then the old link is
        first removed from the dictionary with its name changed to the root
        (``'/'``).

        Args:
            key (str): Relative name of given link.
            value (link): Link object to be set.

        """
        if not isinstance(value, Link):
            raise TypeError('Cannot set a non-link object to DictLink')
        if value._name:
            raise ValueError('Cannot set a link to multiple parents')
        value.name = '%s/%s' % (self._name, key)

        old = self.get(key, None)
        if old is not None:
            old.name = ''
        self.children[key] = value

    def clear(self):
        """Deletes all *child* links in the link dictionary."""
        for link in six.itervalues(self.children):
            link.name = ''
        self.children.clear()

    def get(self, key, *args):
        """Returns the link of given name with default value."""
        return self.children.get(key, *args)

    def items(self):
        """Returns the items of the link dictionary."""
        return self.children.items()

    if sys.version_info.major < 3:
        def iteritems(self):
            """Returns a (key, value) pair iterator of the link dictionary."""
            return self.children.iteritems()

        def iterkeys(self):
            """Returns a key iterator of the link dictionary."""
            return self.children.iterkeys()

        def itervalues(self):
            """Returns a value iterator of the link dictionary."""
            return self.children.itervalues()

    def has_key(self, key):
        """Returns True if given key exists in the link dictionary."""
        return key in self

    def keys(self):
        """Returns the keys of the link dictionary."""
        return self.children.keys()

    def pop(self, key, *args):
        """Removes the link of given key and returns it with default value.

        Note that the name of the returned link is updated to root (``"/"``).

        """
        ret = self.children.pop(key, *args)
        if args and ret is args[0]:
            return ret
        ret.name = ''
        return ret

    def popitem(self):
        """Removes an item randomly and returns it."""
        key, link = self.children.popitem()
        link.name = ''
        return key, link

    def setdefault(self, key, default=None):
        """Gets a link with given name if exists or sets a value."""
        ret = self.children.get(key, None)
        if ret is None:
            self[key] = default
            return default
        else:
            return ret

    def values(self):
        """Returns the values of the link dictionary."""
        return self.children.values()

    @property
    def name(self):
        return self._name or '/'

    @name.setter
    def name(self, name):
        if name == '/':
            name = ''
        self._name = name
        prefix = self._name + '/'
        for key, link in six.iteritems(self):
            link.name = prefix + key

    def copy(self, shared=True):
        ret = super(DictLink, self)._copy_params_states(shared)
        ret.children = {}  # reset children w/o renaming the source ones
        ret.name = ''
        for key, link in six.iteritems(self):
            ret[key] = link.copy(shared)
        return ret

    def visitlinks(self):
        yield self
        for o1 in six.itervalues(self):
            for o2 in o1.visitlinks():
                yield o2

    def serialize(self, serializer):
        Link.serialize(self, serializer)
        for key, link in six.iteritems(self):
            link.serialize(serializer[key])


class ListLink(Link):

    """List-like compostional link.

    ListLink is used to build a compositional link that holds arbitrary number
    of links. It has an interface similar to the Python :class:`list`. The
    values of the "list" are restricted to links, which are considered as
    *child* links. The index of each child link is used as the relative name.

    Args:
        args (tuple): Initial set of links.

    """

    def __init__(self, *args):
        Link.__init__(self)
        for link in args:
            if not isinstance(link, Link):
                raise TypeError('Cannot set a non-link object to ListLink')
            if link._name:
                raise ValueError('Cannot set a link to multiple parents')
        for i, link in enumerate(args):
            link.name = '%s/%d' % (self._name, i)
        self.children = list(args)

    def __getitem__(self, idx):
        """Returns the link at the given index."""
        return self.children[idx]

    def __iter__(self):
        """Returns an iterator of link list."""
        return self.children.__iter__()

    def __len__(self):
        """Returns the number of links in the link list."""
        return len(self.children)

    def __setitem__(self, idx, value):
        """Sets a link to the specified index.

        It also updates the name of the given link to an appropriate path in
        the link hierarchy. The old link at the index is removed from the list
        with its name changed to the root (``'/'``).

        Args:
            idx (int): Index in the link list.
            value (Link): Link object to be set.

        """
        if not isinstance(value, Link):
            raise TypeError('Cannot set a non-link object to ListLink')
        if value._name:
            raise ValueError('Cannot set a link to multiple parents')
        value.name = '%s/%d' % (self._name, idx)

        self.children[idx].name = ''
        self.children[idx] = value

    def append(self, link):
        """Appends a link object to the tail of the link list.

        It also updates the name of the given link to an appropriate path in
        the link hierarchy.

        Args:
            link (Link): Link object to be appended.

        """
        if not isinstance(link, Link):
            raise TypeError('Cannot set a non-link object to ListLink')
        if link._name:
            raise ValueError('Cannot set a link to multiple parents')
        link.name = '%s/%d' % (self._name, len(self.children))
        self.children.append(link)

    def pop(self):
        """Removes the last link in the list and returns it.

        The name of the removed link is updated to the root (``'/'``).

        Returns:
            Link: The last link object in the link list.

        """
        link = self.children.pop()
        link.name = ''
        return link

    @property
    def name(self):
        return self._name or '/'

    @name.setter
    def name(self, name):
        self._name = name
        for i, link in enumerate(self):
            link.name = '%s/%d' % (name, i)

    def copy(self, shared=True):
        ret = super(ListLink, self)._copy_params_states(shared)
        ret.children = []  # reset children w/o renaming the source ones
        ret.name = ''
        for link in self:
            ret.append(link.copy(shared))
        return ret

    def visitlinks(self):
        yield self
        for l1 in self:
            for l2 in l1.visitlinks():
                yield l2

    def serialize(self, serializer):
        Link.serialize(self, serializer)
        for idx, link in enumerate(self):
            link.serialize(serializer[str(idx)])
