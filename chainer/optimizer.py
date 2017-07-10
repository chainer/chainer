import collections
import copy
import warnings

import numpy
import six

from chainer import cuda
from chainer import link as link_module
from chainer import serializer as serializer_module
from chainer import variable


def _sum_sqnorm(arr):
    sq_sum = collections.defaultdict(float)
    for x in arr:
        with cuda.get_device_from_array(x) as dev:
            x = x.ravel()
            s = x.dot(x)
            sq_sum[int(dev)] += s
    return sum([float(i) for i in six.itervalues(sq_sum)])


def exponential_decay_noise(xp, shape, dtype, hook, opt):
    """Time-dependent annealed Gaussian noise function from the paper:

    `Adding Gradient Noise Improves Learning for Very Deep Networks
    <https://arxiv.org/pdf/1511.06807>`_.
    """
    std = numpy.sqrt(hook.eta / numpy.power(1 + opt.t, 0.55))
    return xp.random.normal(0, std, shape).astype(dtype)


class Hyperparameter(object):

    """Set of hyperparameter entries of an optimizer.

    This is a utility class to provide a set of hyperparameter entries for
    update rules and an optimizer. Each entry can be set as an attribute of a
    hyperparameter object.

    A hyperparameter object can hold a reference to its parent hyperparameter
    object. When an attribute does not exist in the child hyperparameter, it
    automatically refers to the parent. We typically set the hyperparameter of
    the gradient method as the parent of the hyperparameter of each update
    rule. It enables us to centralize the management of hyperparameters (e.g.
    we can change the learning rate of all update rules just by modifying the
    hyperparameter of the central optimizer object), while users can freely
    customize the hyperparameter of each update rule if needed.

    Args:
        parent (Hyperparameter): Parent hyperparameter.

    """

    def __init__(self, parent=None):
        self._parent = parent

    def __getattr__(self, name):
        if '_parent' not in self.__dict__:
            raise AttributeError('_parent is not set up yet')
        return getattr(self._parent, name)

    def __repr__(self):
        d = self.get_dict()
        keys = sorted(d.keys())
        values_repr = ', '.join('%s=%s' % (k, d[k]) for k in keys)
        return 'Hyperparameter(%s)' % values_repr

    @property
    def parent(self):
        """Parent hyperparmaeter object."""
        return self._parent

    def get_dict(self):
        """Converts the hyperparameter into a dictionary.

        Returns:
            Dictionary containing all entries that can be referred by this
            hyperparameter object.

        """
        d = {} if self._parent is None else self._parent.get_dict()
        for k, v in six.iteritems(self.__dict__):
            if k != '_parent':
                d[k] = v
        return d


class UpdateRule(object):

    """Base class of all update rules.

    Update rule is an object that implements how to update one parameter
    variable using the gradient of a loss function. This class provides the
    interface and the common features of any update rules.

    An update rule can be set to a :class:`~chainer.Variable` object that
    represents a parameter array of a model. An :class:`~chainer.Optimizer`
    instance defines which parameters to update, and the update rule instance
    of each parameter defines how to update it.

    Hook functions can be set to any update rule instance. The hook function is
    called just before any updates in the order of registrations.

    An implementation of update rule should override :meth:`update_core` or
    its device-dependent variants (i.e., :meth:`update_core_cpu` and
    :meth:`update_core_gpu`).

    The state (e.g. a moving average of the gradient) of the update rule is
    stored into the state dictionary. An implementation of update rule using
    state should also override :meth:`init_state` to initialize the state at
    the first update. The values of the state dictionary are automatically
    copied to the appropriate device before the update based on the data and
    grad arrays.

    Args:
        parent_hyperparam (Hyperparameter): Hyperparameter that provides the
            default values.

    Attributes:
        enabled (bool): Flag to configure if this update rule is active. If the
            update rule is not active (i.e., ``enabled = False``), the
            :meth:`update` method does not update the parameter.
        hyperparam (Hyperparameter): Hyperparameter of the update rule.
        t (int): Number of updates made by this update rule.

    """

    def __init__(self, parent_hyperparam=None):
        self._hooks = collections.OrderedDict()
        self._state = None
        self.enabled = True
        self.hyperparam = Hyperparameter(parent_hyperparam)
        self.t = 0

    @property
    def state(self):
        """State dictionary."""
        return self._state

    def add_hook(self, hook, name=None):
        """Adds a hook function.

        The hook function is called before any updates.

        Args:
            hook (callable): Hook function to be added. It takes two
                arguments: the update rule object and the parameter variable.
            name (str): Name of the hook function. The name attribute of the
                hook function is used by default.

        """
        if not callable(hook):
            raise TypeError('hook function must be callable')

        if name is None:
            name = getattr(hook, 'name', getattr(hook, '__name__', None))
            if name is None:
                raise ValueError(
                    'the name of the hook function is not specified')
        if name in self._hooks:
            raise ValueError('hook "{}" already exists'.format(name))

        self._hooks[name] = hook

    def remove_hook(self, name):
        """Removes the specified hook function.

        Args:
            name (str): Name of the hook function to be removed. The hook
                function registered with this name will be removed.

        """
        del self._hooks[name]

    def update(self, param):
        """Invokes hook functions and updates the parameter.

        Args:
            param (~chainer.Variable): Variable to be updated.

        """
        if not self.enabled:
            return

        self.t += 1
        self._prepare(param)
        for hook in six.itervalues(self._hooks):
            hook(self, param)
        self.update_core(param)

    def update_core(self, param):
        """Updates the parameter.

        Implementation of UpdateRule should override this method or both of
        :meth:`_update_core_cpu` and :meth:`_update_core_gpu`.

        Args:
            param (~chainer.Variable): Variable to be updated.

        """
        with cuda.get_device_from_array(param.data) as dev:
            if int(dev) == -1:
                self.update_core_cpu(param)
            else:
                self.update_core_gpu(param)

    def update_core_cpu(self, param):
        """Updates the parameter on CPU.

        See :meth:`update_core` for details.

        Args:
            param (~chainer.Variable): Variable to be updated.

        """
        raise NotImplementedError

    def update_core_gpu(self, param):
        """Updates the parameter on GPU.

        See :meth:`update_core` for details.

        Args:
            param (~chainer.Variable): Variable to be updated.

        """
        raise NotImplementedError

    def init_state(self, param):
        """Initializes the state.

        Any implementations that use the state should override this mehtod.
        This method is called at the first update.

        Args:
            param (~chainer.Variable): Parameter variable. It can be used to
                extract the shape and the data type of the parameter.

        """
        pass

    def serialize(self, serializer):
        """Serializes the update rule state.

        Be careful that this method only saves/loads the state of the update
        rule. The parameters of the target link is not saved/loaded by this
        method, and so you need to serialize the target link separately if you
        want to fully recover the training state including parameters.

        Args:
            serializer (~chainer.AbstractSerializer): Serializer object.

        """
        if self.state is None:
            if isinstance(serializer, serializer_module.Deserializer):
                # try to initialize the state to retrieve state entries
                self._state = {}
                self_copy = copy.copy(self)
                arr = numpy.empty(1, dtype=numpy.float32)
                self_copy.init_state(variable.Variable(arr, grad=arr))

                for key in self._state:
                    self._state[key] = serializer(key, None)
        else:
            for key in self._state:
                self._state[key] = serializer(key, self._state[key])

    def _prepare(self, param):
        with cuda.get_device_from_array(param.data) as device:
            state = self.state
            if state is None:
                state = self._state = {}
                self.init_state(param)

            for name, value in six.iteritems(state):
                if not isinstance(value, (numpy.ndarray, cuda.ndarray)):
                    continue
                value_device = cuda.get_device_from_array(value)
                if value_device.id != device.id:
                    if device.id >= 0:
                        state[name] = cuda.to_gpu(value)
                    else:
                        state[name] = cuda.to_cpu(value)


class Optimizer(object):
    """Base class of all numerical optimizers.

    This class provides basic features for all optimization methods. It
    optimizes parameters of a *target link*. The target link is registered via
    the :meth:`setup` method, and then the :meth:`update` method updates its
    parameters based on a given loss function.

    Each optimizer implementation must be defined as a child class of
    Optimizer. It must override :meth:`update` method.

    If the optimizer is based on single gradient computation (like
    most first-order methods), then it should inherit :class:`GradientMethod`,
    which adds some features dedicated for the first order methods, including
    the support of :class:`~chainer.UpdateRule`.

    Optimizer instance also supports *hook functions*. Hook function is
    registered by the :meth:`add_hook` method. Each hook function is called
    in registration order in advance of the actual parameter update. If the
    hook function has an attribute ``call_for_each_param`` and its value is
    ``True``, the hook function is used as a hook function of all update rules
    (i.e., it is invoked for every parameter by passing the corresponding
    update rule and the parameter).

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
        self._hooks = collections.OrderedDict()

    def update(self, lossfun=None, *args, **kwds):
        """Updates the parameters.

        This method updates the parameters of the target link. The behavior of
        this method is different for the cases either ``lossfun`` is given or
        not.

        If ``lossfun`` is given, this method typically clears the gradients,
        calls the loss function with given extra arguments, and calls the
        :meth:`~chainer.Variable.backward` method of its output to compute the
        gradients. The actual implementation might call ``lossfun`` more than
        once.

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
            hook (function): Hook function. If ``hook.call_for_each_param`` is
                true, this hook function is called for each parameter by
                passing the update rule and the parameter. Otherwise, this hook
                function is called only once each iteration by passing the
                optimizer.
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
            self._call_hook(hook)

    def _call_hook(self, hook):
        if getattr(hook, 'call_for_each_param', False):
            for param in self.target.params():
                hook(param.update_rule, param)
        else:
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
        for name, param in self.target.namedparams():
            rule = getattr(param, 'update_rule', None)
            if rule is not None:
                rule.serialize(serializer[name])


class GradientMethod(Optimizer):
    """Base class of all single gradient-based optimizers.

    This is an extension of the :class:`Optimizer` class. Typical gradient
    methods that just require the gradient at the current parameter vector on
    an update can be implemented as its child class.

    This class uses :class:`~chainer.UpdateRule` to manage the update rule of
    each parameter. A child class of GradientMethod should override
    :meth:`create_update_rule` to create the default update rule of each
    parameter.

    This class also provides :attr:`hyperparam`, which is the hyperparameter
    used as the default configuration of each update rule. All built-in
    gradient method implementations also provide proxy properties that act
    as aliases to the attributes of :attr:`hyperparam`. It is recommended to
    provide such an alias to each attribute. It can be done by only adding one
    line for each attribute using :class:`HyperparameterProxy`.

    Attributes:
        hyperparam (Hyperparameter): The hyperparameter of the gradient
            method. It is used as the default configuration of each update
            rule (i.e., the hyperparameter of each update rule refers this
            hyperparameter as its parent).

    """

    def __init__(self):
        super(GradientMethod, self).__init__()
        self.hyperparam = Hyperparameter()

    def setup(self, link):
        super(GradientMethod, self).setup(link)
        for param in link.params():
            param.update_rule = self.create_update_rule()

    def reallocate_cleared_grads(self):
        """Reallocate gradients cleared by :meth:`~chainer.Variable.cleargrad`.

        This method allocates arrays for all gradients which have :obj:`None`.
        This method is called before and after every optimizer hook.
        If an inheriting optimizer does not require this allocation,
        the optimizer can override this method with a blank function.

        """
        for name, param in self.target.namedparams(False):
            if param.grad is None:
                with cuda.get_device_from_array(param.data):
                    xp = cuda.get_array_module(param.data)
                    param.grad = xp.zeros_like(param.data)

    def call_hooks(self):
        """Invokes hook functions in registration order."""
        for hook in six.itervalues(self._hooks):
            self._call_hook(hook)
            self.reallocate_cleared_grads()

    def update(self, lossfun=None, *args, **kwds):
        """Updates parameters based on a loss function or computed gradients.

        This method runs in two ways.

        - If ``lossfun`` is given, then it is used as a loss function to
          compute gradients.
        - Otherwise, this method assumes that the gradients are already
          computed.

        In both cases, the computed gradients are used to update parameters.
        The actual update routines are defined by the update rule of each
        parameter.

        """
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            loss.backward()
            del loss

        self.reallocate_cleared_grads()

        self.call_hooks()

        self.t += 1
        for param in self.target.params():
            param.update()

    def use_cleargrads(self, use=True):
        """Enables or disables use of :func:`~chainer.Link.cleargrads` in `update`.

        Args:
            use (bool): If ``True``, this function enables use of
                `cleargrads`. If ``False``, disables use of `cleargrads`
                (`zerograds` is used).

        .. deprecated:: v2.0
           Note that :meth:`update` calls :meth:`~Link.cleargrads` by default.
           :meth:`~Link.cleargrads` is more efficient than
           :meth:`~Link.zerograds`, so one does not have to call
           :meth:`use_cleargrads`. This method remains for backward
           compatibility.

        """
        warnings.warn(
            'GradientMethod.use_cleargrads is deprecated.',
            DeprecationWarning)

        self._use_cleargrads = use

    def create_update_rule(self):
        """Creates a new update rule object.

        This method creates an update rule object. It is called by
        :meth:`setup` to set up an update rule of each parameter.
        Each implementation of the gradient method should override this method
        to provide the default update rule implementation.

        Return:
            UpdateRule: Update rule object.

        """
        raise NotImplementedError


class HyperparameterProxy(object):

    """Property that acts as an alias to an attribute of the hyperparameter.

    This class is used to define a property of an implementation of
    :class:`GradientMethod` that acts as an alias to an attribute of the
    hyperparameter.

    Args:
        attr_name (str): Name of the attribute of the hyperparameter.

    """

    def __init__(self, attr_name):
        self._attr_name = attr_name
        self.__doc__ = 'Alias to ``self.hyperparam.{}``'.format(attr_name)

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        return getattr(obj.hyperparam, self._attr_name)

    def __set__(self, obj, value):
        setattr(obj.hyperparam, self._attr_name, value)


class WeightDecay(object):

    """Optimizer/UpdateRule hook function for weight decay regularization.

    This hook function adds a scaled parameter to the corresponding gradient.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.

    Attributes:
        rate (float): Coefficient for the weight decay.

    """
    name = 'WeightDecay'
    call_for_each_param = True

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        p, g = param.data, param.grad
        with cuda.get_device_from_array(p) as dev:
            if int(dev) == -1:
                g += self.rate * p
            else:
                kernel = cuda.elementwise(
                    'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')
                kernel(p, self.rate, g)


class Lasso(object):
    """Optimizer/UpdateRule hook function for Lasso regularization.

    This hook function adds a scaled parameter to the sign of each weight.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.

    Attributes:
        rate (float): Coefficient for the weight decay.

    """
    name = 'Lasso'
    call_for_each_param = True

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        p, g = param.data, param.grad
        xp = cuda.get_array_module(p)
        with cuda.get_device_from_array(p) as dev:
            sign = xp.sign(p)
            if int(dev) == -1:
                g += self.rate * sign
            else:
                kernel = cuda.elementwise(
                    'T s, T decay', 'T g', 'g += decay * s', 'lasso')
                kernel(sign, self.rate, g)


class GradientClipping(object):
    """Optimizer hook function for gradient clipping.

    This hook function scales all gradient arrays to fit to the defined L2 norm
    threshold.

    Args:
        threshold (float): L2 norm threshold.

    Attributes:
        threshold (float): L2 norm threshold of gradient norm.

    """
    name = 'GradientClipping'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        norm = numpy.sqrt(_sum_sqnorm(
            [p.grad for p in opt.target.params(False)]))
        rate = self.threshold / norm
        if rate < 1:
            for param in opt.target.params(False):
                grad = param.grad
                with cuda.get_device_from_array(grad):
                    grad *= rate


class GradientNoise(object):
    """Optimizer/UpdateRule hook function for adding gradient noise.

    This hook function simply adds noise generated by the ``noise_func``
    to the gradient. By default it adds time-dependent annealed Gaussian
    noise to the gradient at every training step:

    .. math::

        g_t \\leftarrow g_t + N(0, \\sigma_t^2)

    where

    .. math::

        \\sigma_t^2 = \\frac{\\eta}{(1+t)^\\gamma}

    with :math:`\\eta` selected from {0.01, 0.3, 1.0} and
    :math:`\\gamma = 0.55`.

    Args:
        eta (float): Parameter that defines the scale of the noise, which for
            the default noise function is recommended to be either 0.01, 0.3
            or 1.0.
        noise_func (function): Noise generating function which by default
            is given by `Adding Gradient Noise Improves Learning for Very Deep\
            Networks <https://arxiv.org/pdf/1511.06807>`_.
    """
    name = 'GradientNoise'
    call_for_each_param = True

    def __init__(self, eta, noise_func=exponential_decay_noise):
        self.eta = eta
        self.noise_func = noise_func

    def __call__(self, rule, param):
        g = param.grad
        xp = cuda.get_array_module(g)
        with cuda.get_device_from_array(g) as dev:
            noise = self.noise_func(xp, g.shape, g.dtype, self, rule)
            if int(dev) == -1:
                g += noise
            else:
                kernel = cuda.elementwise(
                    'T noise', 'T g', 'g += noise', 'gradient_noise')
                kernel(noise, g)


class GradientHardClipping(object):

    """Optimizer/UpdateRule hook function for gradient clipping.

    This hook function clips all gradient arrays to be within a lower and upper
    bound.

    Args:
        lower_bound (float): The lower bound of the gradient value.
        upper_bound (float): The upper bound of the gradient value.

    Attributes:
        lower_bound (float): The lower bound of the gradient value.
        upper_bound (float): The upper bound of the gradient value.

    """
    name = 'GradientHardClipping'
    call_for_each_param = True

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, rule, param):
        grad = param.grad
        xp = cuda.get_array_module(grad)
        with cuda.get_device_from_array(grad):
            xp.clip(grad, self.lower_bound, self.upper_bound, out=grad)
