import collections
import sys
import warnings

import numpy
import six

from chainer import cuda
import chainer.link as link_module
from chainer import optimizer_hooks


def _sum_sqnorm(arr):
    sq_sum = collections.defaultdict(float)
    for x in arr:
        with cuda.get_device(x) as dev:
            x = x.ravel()
            s = x.dot(x)
            sq_sum[int(dev)] += s
    return sum([float(i) for i in six.itervalues(sq_sum)])


class Optimizer(object):
    """Base class of all numerical optimizers.

    This class provides basic features for all optimization methods. It
    optimizes parameters of a *target link*. The target link is registered via
    the :meth:`setup` method, and then the :meth:`update` method updates its
    parameters based on a given loss function.

    Each optimizer implementation must be defined as a child class of
    Optimizer. It must override :meth:`update` method. An optimizer can use
    *internal states* each of which is tied to one of the parameters. State is
    a dictionary of serializable values (typically arrays of size same as
    the corresponding parameters). In order to use state dictionaries, the
    optimizer must override :meth:`init_state` method (or its CPU/GPU versions,
    :meth:`init_state_cpu` and :meth:`init_state_gpu`).

    If the optimizer is based on single gradient computation (like
    most first-order methods), then it should inherit :class:`GradientMethod`,
    which adds some features dedicated for the first order methods.

    Optimizer instance also supports *hook functions*. Hook function is
    registered by the :meth:`add_hook` method. Each hook function is called
    in registration order in advance of the actual parameter update.

    Attributes:
        target: Target link object. It is set by the :meth:`setup` method.
        t: Number of update steps. It must be incremented by the
            :meth:`update` method.
        epoch: Current epoch. It is incremented by the :meth:`new_epoch`
            method.

    """

    def setup(self, link):
        """Sets a target link and initializes the optimizer states.

        Given link is set to the :attr:`target` attribute. It also prepares the
        optimizer state dictionaries corresponding to all parameters in the
        link hierarchy. The existing states are discarded.

        Args:
            link (~chainer.Link): Target link object.

        """
        if not isinstance(link, link_module.Link):
            raise TypeError('optimization target must be a link')
        self.target = link
        self.t = 0
        self.epoch = 0
        self._states = {}
        self._hooks = collections.OrderedDict()

        self.prepare()

    def prepare(self):
        """Prepares for an update.

        This method initializes missing optimizer states (e.g. for newly added
        parameters after the set up), and copies arrays in each state
        dictionary to CPU or GPU according to the corresponding parameter
        array.

        """
        states = self._states
        for name, param in self.target.namedparams():
            if name not in states:
                state = {}
                self.init_state(param, state)
                states[name] = state
            else:
                state = states[name]
                with cuda.get_device(param.data) as dev:
                    if int(dev) == -1:  # cpu
                        for key, value in six.iteritems(state):
                            if isinstance(value, cuda.ndarray):
                                state[key] = value.get()
                    else:  # gpu
                        cupy = cuda.cupy
                        for key, value in six.iteritems(state):
                            if isinstance(value, numpy.ndarray):
                                state[key] = cuda.to_gpu(value)
                            elif (isinstance(value, cupy.ndarray) and
                                  value.device != dev):
                                state[key] = cupy.copy(value)

    def init_state(self, param, state):
        """Initializes the optimizer state corresponding to the parameter.

        This method should add needed items to the ``state`` dictionary. Each
        optimizer implementation that uses its own states should override this
        method or CPU/GPU dedicated versions (:meth:`init_state_cpu` and
        :meth:`init_state_gpu`).

        Args:
            param (~chainer.Variable): Parameter variable.
            state (dict): State dictionary.

        .. seealso:: :meth:`init_state_cpu`, :meth:`init_state_gpu`

        """
        with cuda.get_device(param.data) as dev:
            if int(dev) == -1:
                self.init_state_cpu(param, state)
            else:
                self.init_state_gpu(param, state)

    def init_state_cpu(self, param, state):
        """Initializes the optimizer state on CPU.

        This method is called from :meth:`init_state` by default.

        Args:
            param (~chainer.Variable): Parameter variable. Its data array is
                of type :class:`numpy.ndarray`.
            state (dict): State dictionary.

        .. seealso:: :meth:`init_state`

        """
        pass

    def init_state_gpu(self, param, state):
        """Initializes the optimizer state on GPU.

        This method is called from :meth:`init_state` by default.

        Args:
            param (~chainer.Variable): Parameter variable. Its data array is
                of type :class:`cupy.ndarray`.
            state (dict): State dictionary.

        .. seealso:: :meth:`init_state`

        """
        pass

    def update(self, lossfun=None, *args, **kwds):
        """Updates the parameters and optimizer states.

        This method updates the parameters of the target link and corresponding
        optimizer states. The behavior of this method is different for the
        cases either ``lossfun`` is given or not.

        If ``lossfun`` is given, then this method initializes the gradients by
        zeros, calls it with given extra arguments, and calls the
        :meth:`~chainer.Variable.backward` method of its output to compute the
        gradients. The implementation might call ``lossfun`` more than once.

        If ``lossfun`` is not given, then this method assumes that the
        gradients of all parameters are already computed. An implementation
        that requires multiple gradient computations might raise an error on
        this case.

        In both cases, this method invokes the update procedure for all
        parameters.

        Args:
            lossfun (function): Loss function. It accepts arbitrary arguments
                and returns one :class:`~chainer.Variable` object that
                represents the loss (or objective) value. This argument can be
                omitted for single gradient-based methods. In this case, this
                method assumes gradient arrays computed.
            args, kwds: Arguments for the loss function.

        """
        raise NotImplementedError

    def new_epoch(self):
        """Starts a new epoch.

        This method increments the :attr:`epoch` count. Note that if the
        optimizer depends on the epoch count, then user should call this method
        appropriately at the beginning of each epoch.

        """
        self.epoch += 1

    def add_hook(self, hook, name=None):
        """Registers a hook function.

        Hook function is typically called right after the gradient computation,
        though the timing depends on the optimization method.

        Args:
            hook (function): Hook function. It accepts the optimizer object.
            name (str): Name of the registration. If omitted, ``hook.name`` is
                used by default.

        """
        if not callable(hook):
            raise TypeError('hook function is not callable')
        if not hasattr(self, '_hooks'):
            raise RuntimeError('call `setup` method before `add_hook` method')

        if name is None:
            name = hook.name
        if name in self._hooks:
            raise KeyError('hook %s already exists' % name)
        self._hooks[name] = hook

    def remove_hook(self, name):
        """Removes a hook function.

        Args:
            name (str): Registered name of the hook function to remove.

        """
        del self._hooks[name]

    def call_hooks(self):
        """Invokes hook functions in registration order."""
        for hook in six.itervalues(self._hooks):
            hook(self)

    def serialize(self, serializer):
        """Serializes or deserializes the optimizer.

        It only saves or loads the following things:

        - Optimizer states
        - Global states (:attr:`t` and :attr:`epoch`)

        **It does not saves nor loads the parameters of the target link.** They
        should be separately saved or loaded.

        Args:
            serializer (~chainer.AbstractSerializer): Serializer or
                deserializer object.

        """
        self.t = serializer('t', self.t)
        self.epoch = serializer('epoch', self.epoch)
        for name, state in six.iteritems(self._states):
            s = serializer[name]
            for key, value in six.iteritems(state):
                state[key] = s(key, value)

    def zero_grads(self):
        """Fills all gradient arrays by zeros.

        .. deprecated:: v1.5
           Use the :meth:`chainer.Link.cleargrads` method for the target link
           instead.

        """
        self.target.zerograds()

    def compute_grads_norm(self):
        """Computes the norm of whole gradients.

        Returns:
            float: L2 norm of whole gradients, i.e. square root of sum of
            square of all gradient elements.

        .. warning::

            This method returns a CPU-computed value, which means that this
            method synchronizes between CPU and GPU if at least one of the
            gradients reside on the GPU.

        .. deprecated:: v1.5

        """
        return numpy.sqrt(_sum_sqnorm([p.grad for p in self.target.params()]))

    def clip_grads(self, maxnorm):
        """Clips the norm of whole gradients up to the threshold.

        Args:
            maxnorm (float): Threshold of gradient L2 norm.

        .. deprecated:: v1.5
           Use the :class:`~chainer.optimizer_hooks.GradientClipping`
           hook function instead.

        """
        optimizer_hooks.GradientClipping(maxnorm)(self)

    def weight_decay(self, decay):
        """Applies weight decay to the parameter/gradient pairs.

        Args:
            decay (float): Coefficient of weight decay.

        .. deprecated:: v1.5
           Use the :class:`~chainer.optimizer.WeightDecay` hook function
           instead.

        """
        optimizer_hooks.WeightDecay(decay)(self)

    def accumulate_grads(self, grads):
        """Accumulates gradients from other source.

        This method just adds given gradient arrays to gradients that this
        optimizer holds. It is typically used in data-parallel optimization,
        where gradients for different shards are computed in parallel and
        aggregated by this method. This method correctly treats multiple GPU
        devices.

        Args:
            grads (Iterable): Iterable of gradient arrays to be accumulated.

        .. deprecated:: v1.5
           Use the :meth:`chainer.Link.addgrads` method of the target link
           instead.

        """
        for param, g_src in zip(self.target.params(), grads):
            g_dst = param.grad
            if isinstance(g_dst, numpy.ndarray):
                g_dst += cuda.to_cpu(g_src)
                continue

            with cuda.get_device(g_dst):
                if (isinstance(g_src, cuda.ndarray) and
                        g_dst.device != g_src.device):
                    g_dst += cuda.copy(g_src, out_device=g_dst.device)
                else:
                    g_dst += cuda.to_gpu(g_src)


class GradientMethod(Optimizer):
    """Base class of all single gradient-based optimizers.

    This is an extension of the :class:`Optimizer` class. Typical gradient
    methods that just require the gradient at the current parameter vector on
    an update can be implemented as its child class.

    An implementation of a gradient method must override the following methods:

    - :meth:`init_state` or both :meth:`init_state_cpu` and
      :meth:`init_state_gpu`
    - :meth:`update_one` or both :meth:`update_one_cpu` and
      :meth:`update_one_gpu`

    .. note::
       It is recommended to call :meth:`use_cleargrads` after creating a
       :class:`GradientMethod` object for efficiency.

    """

    def reallocate_cleared_grads(self):
        """Reallocate gradients cleared by :meth:`~chainer.Variable.cleargrad`.

        This method allocates arrays for all gradients which have :obj:`None`.
        This method is called before and after every optimizer hook.
        If an inheriting optimizer does not require this allocation,
        the optimizer can override this method with a blank function.

        """
        for name, param in self.target.namedparams():
            if param.grad is None:
                with cuda.get_device(param.data):
                    xp = cuda.get_array_module(param.data)
                    param.grad = xp.zeros_like(param.data)

    def call_hooks(self):
        """Invokes hook functions in registration order."""
        for hook in six.itervalues(self._hooks):
            hook(self)
            self.reallocate_cleared_grads()

    def update(self, lossfun=None, *args, **kwds):
        """Updates parameters based on a loss function or computed gradients.

        This method runs in two ways.

        - If ``lossfun`` is given, then use it as a loss function to compute
          gradients.
        - Otherwise, this method assumes that the gradients are already
          computed.

        In both cases, the computed gradients are used to update parameters.
        The actual update routines are defined by the :meth:`update_one`
        method (or its CPU/GPU versions, :meth:`update_one_cpu` and
        :meth:`update_one_gpu`).

        """
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', False)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            loss.backward()
            del loss

        self.reallocate_cleared_grads()

        self.call_hooks()
        self.prepare()

        self.t += 1
        states = self._states
        for name, param in self.target.namedparams():
            with cuda.get_device(param.data):
                self.update_one(param, states[name])

    def update_one(self, param, state):
        """Updates a parameter based on the corresponding gradient and state.

        This method calls appropriate one from :meth:`update_param_cpu` or
        :meth:`update_param_gpu`.

        Args:
            param (~chainer.Variable): Parameter variable.
            state (dict): State dictionary.

        """
        if isinstance(param.data, numpy.ndarray):
            self.update_one_cpu(param, state)
        else:
            self.update_one_gpu(param, state)

    def update_one_cpu(self, param, state):
        """Updates a parameter on CPU.

        Args:
            param (~chainer.Variable): Parameter variable.
            state (dict): State dictionary.

        """
        raise NotImplementedError

    def update_one_gpu(self, param, state):
        """Updates a parameter on GPU.

        Args:
            param (~chainer.Variable): Parameter variable.
            state (dict): State dictionary.

        """
        raise NotImplementedError

    def use_cleargrads(self, use=True):
        """Enables or disables use of :func:`~chainer.Link.cleargrads` in `update`.

        Args:
            use (bool): If ``True``, this function enables use of
                `cleargrads`. If ``False``, disables use of `cleargrads`
                (`zerograds` is used).

        .. note::
           Note that :meth:`update` calls :meth:`~Link.zerograds` by default
           for backward compatibility. It is recommended to call this method
           before first call of `update` because `cleargrads` is more
           efficient than `zerograds`.

        """
        self._use_cleargrads = use


def DeprecationWarningWrapper(mod, deprecated):

    deprecated = set(deprecated)

    class Wrapper(object):

        def __getattr__(self, attr):
            if attr in deprecated:
                warnings.warn(
                    "chainer.optimizer.{0} is deprecated since v1.23.0. "
                    "Use chainer.optimizer_hooks.{0} instead.".format(attr),
                    DeprecationWarning)

            return getattr(mod, attr)

        def __setattr__(self, attr, value):
            if attr in deprecated:
                warnings.warn(
                    "chainer.optimizer.{0} is deprecated since v1.23.0. "
                    "Use chainer.optimizer_hooks.{0} instead".format(attr),
                    DeprecationWarning)
            return setattr(mod, attr, value)
    return Wrapper()


#  Backward compatibility
GradientClipping = optimizer_hooks.GradientClipping
GradientHardClipping = optimizer_hooks.GradientHardClipping
GradientNoise = optimizer_hooks.GradientNoise
Lasso = optimizer_hooks.Lasso
WeightDecay = optimizer_hooks.WeightDecay

sys.modules[__name__] = DeprecationWarningWrapper(
    sys.modules[__name__],
    deprecated=['GradientClipping',
                'GradientHardClipping',
                'GradientNoise',
                'Lasso',
                'WeightDecay'])
