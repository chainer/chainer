import math
import warnings

import numpy
import six

from chainer import cuda
from chainer import link
from chainer import variable


class TupleLink(link.Link):

    def __init__(self, params_grads):
        super(TupleLink, self).__init__()
        for i, (param, grad) in enumerate(six.moves.zip(*params_grads)):
            var = variable.Variable(param)
            var.grad = grad
            self.params[str(i)] = var


class Optimizer(object):

    """Base class of all numerical optimizers.

    Optimizer is set up with a target link whose parameters are to be
    optimized. It is done by passing the target link to the :meth:`setup`
    method. After that, the :meth:`update` method updates the parameters.

    Each optimizer implementation can create *state values* for each
    parameter variable of the target link. The state values of the optimizer
    can be saved via the Chainer's serialization protocol.

    Many optimizers are implemented under the :mod:`optimizers` subpackage.

    Attributes:
        t (int): Number of updates done (a.k.a. iterations).
        target (Link): The target link. It is set by the :meth:`setup` method.
        states (dict): Dictionary of state arrays. Each state corresponds to
            one parameter, and the key of the state is the absolute name of the
            parameter in the target link.

    """
    def __init__(self):
        self.t = 0
        self.target = None
        self.states = {}

    def init_state(self, param, state):
        """Initializes the state values for given parameter array.

        State values are represented by a dictionary with string keys and array
        values. State arrays must be on the same device as the parameter array.

        Optimizer implementation should override this method if it requires
        states to be maintained accross multiple updates. The default
        implementation just calls the :meth:`init_state_cpu` or
        :meth:`init_state_gpu` method depending on the given parameter.

        Args:
            param (array): Parameter array.
            state (dict): State dictionary to be set up at this method.

        """
        with cuda.get_device(param) as d:
            if int(d) < 0:
                self.init_state_cpu(param, state)
            else:
                self.init_state_gpu(param, state)

    def init_state_cpu(self, param, state):
        """Initializes the state values for given parameter on CPU.

        Optimizer implementation can override this method and the
        :meth:`init_state_gpu` instead of the :meth:`init_state` method.

        """
        pass

    def init_state_gpu(self, param, state):
        """Initializes the state values for given parameter on GPU.

        Optimizer implementation can override this method and the
        :meth:`init_state_cpu` instead of the :meth:`init_state` method.

        """
        pass

    def setup(self, target):
        """Sets up the optimizer with given target link.

        It registers the given link to this optimizer. The optimizer holds a
        reference to the link, which is used by other methods.

        Args:
            target (Link): Target link to be optimized.

        """
        if isinstance(target, tuple):
            target = TupleLink(target)
        self.target = target
        for path, param in target.visitparams():
            state = {}
            with cuda.get_device(param.data):
                self.init_state(param.data, state)
            self.states[path] = state

    def update(self, loss_func=None, *args, **kwds):
        """Updates the parameters of the target link.

        .. note::
           This is an abstract method. Implementation class must override it.

        Given a loss function that returns an output :class:`Variable` object,
        this method updates the parameters of the target link.

        If the optimizer implementation allows the gradients to be computed
        outside of this method, then the loss function can be omitted (i.e.
        None), where this method assumes that the gradient arrays has been
        already computed. This is valid for optimizers that inherit
        :class:`GradientMethod`.

        Args:
            loss_func (callable): A callable object that returns a loss
                :class:`Variable` object. The loss variable is used for
                backward computations.
            args, kwds: Arguments of the loss function.

        """
        raise NotImplementedError

    def zero_grads(self):
        """Initializes the gradient arrays of the target link by zeros.

        .. deprecated:: v1.4
           Use :meth:`Link.zerograds` instead.

        """
        warnings.warn('Optimizer.zero_grads is deprecated. '
                      'Use Link.zerograds instead.', DeprecationWarning)
        self.target.zerograds()

    def compute_grads_norm(self):
        """Computes the L2 norm of the whole gradient array.

        .. deprecated:: v1.4
           This method is deprecated.

        """
        warnings.warn('Optimizer.compute_grads_norm is deprecated.',
                      DeprecationWarning)
        sqnorm = 0
        for _, param in self.target.visitparams():
            grad = param.grad.ravel()
            sqnorm += float(grad.dot(grad))
        return math.sqrt(sqnorm)

    def clip_grads(self, maxnorm):
        """Clips the whole gradient array to given L2 norm.

        .. deprecated:: v1.4
           Use the :class:`optimizer.GradientClipping` hook instead.

        """
        warnings.warn('Optimizer.clip_grads is deprecated. '
                      'Use GradientClipping instead.', DeprecationWarning)
        GradientClipping(maxnorm)(self.target)

    def weight_decay(self, decay):
        """Applies weight decay to the parameters and gradients.

        .. deprecated:: v1.4
           Use the :class:`optimizer.WeightDecay` hook instead.

        """
        warnings.warn('Optimizer.weight_decay is deprecated. '
                      'Use WeightDecay instead.', DeprecationWarning)
        WeightDecay(decay)(self.target)

    def accumulate_grads(self, grads):
        """Accumulates given gradients to the gradients of the target link.

        .. deprecated:: v1.4
           Use the :meth:`Link.addgrads` method instead.

        """
        warnings.warn('Optimizer.accumulate_grads is deprecated. '
                      'Use Link.addgrads instead.',
                      DeprecationWarning)
        paths = []
        for path, _ in self.target.visitparams():
            paths.append(path)
        paths.sort()
        d = dict(six.moves.zip(paths, grads))

        for path, param in self.target.visitparams():
            dst = param.grad
            src = d[path]
            if isinstance(dst, numpy.ndarray):
                dst += cuda.to_cpu(src)
            elif isinstance(src, numpy.ndarray):
                dst += cuda.to_gpu(src, device=dst)
            elif src.device == dst.device:
                dst += src
            else:
                dst += cuda.copy(src, out_device=dst)

    def serialize(self, serializer):
        """Serializes states of the optimizer by a given serializer.

        Note that this method does not (de)serialize the target link. It should
        be saved/loaded separately from the optimizer.

        Args:
            serializer (Serializer): Serializer object.

        """
        self.t = serializer('t', self.t)
        serializer = serializer['state']
        for path, state in six.iteritems(self.states):
            s = serializer[path[1:]]  # omit '/' at the head
            for key in state:
                state[key] = s(key, state[key])


class GradientMethod(Optimizer):

    """Base class of first order optimizers based on one gradient per update.

    This class implements a general gradient method procedure.

    In addition to the common interface of :class:`Optimizer`, GradientMethod
    also supports *hook functions*. Once a hook function is registered by
    :meth:`~GradientMethod.add_hook`, it is automatically called by the
    :meth:`~GradientMethod.update` method just before the actual update process
    runs. Hook functions are called in the order of registerations.

    """
    def __init__(self):
        Optimizer.__init__(self)
        self.hooks = []

    def add_hook(self, name, hook):
        """Adds a hook function.

        The hook function takes the target link as an argument and modifies its
        parameters and gradients.

        Args:
            name (str): Name for the registeration. It is used for removing the
                hook.
            hook (callable): Hook function.

        """
        if not callable(hook):
            raise TypeError('Cannot set non-callable object as a hook')
        self.hooks.append((name, hook))

    def remove_hook(self, name):
        """Removes a hook function of given registeration name.

        Args:
            name (str): Name for the registeration.

        """
        dels = []
        for i, (n, h) in enumerate(self.hooks):
            if n == name:
                dels.append(i)
        dels.reverse()
        for i in dels:
            del self.hooks[i]

    def update(self, loss_func=None, *args, **kwds):
        """Updates the parameters with a loss function or computed gradients.

        If a loss function is given, then this method uses it to get a loss
        :class:`Variable` object, and calls its :meth:`~Variable.backward`
        method to compute the gradient arrays. In this case, this method
        initializes the gradients by zero, so users do not have to call
        :meth:`Link.zerograds` beforehand.

        If a loss function is not given, this method assumes that all gradient
        arrays for the parameters of the target link have been already
        computed.

        This method also calls registered hook functions in the order of
        registerations just after the gradient computation.

        Each gradient method implementation should not override this method.
        Instead, :meth:`update_param` or both
        :meth:`update_param_cpu` and :meth:`update_param_gpu` methods should be
        overridden.

        Args:
            loss_func (callable): A callable object that returns a loss
                :class:`Variable` object. The loss variable is used for
                backward computation. If the loss function is not given, then
                this method assumes that the gradient arrays have been already
                computed, and uses them in the update.
            args, kwds: Arguments of the loss function.

        """
        if loss_func is not None:
            self.target.zerograds()
            loss = loss_func(*args, **kwds)
            loss.backward()

        for _, hook in self.hooks:
            hook(self.target)

        self.t += 1
        for path, param in self.target.visitparams():
            state = self.states[path]
            self.update_param(param.data, param.grad, state)

    def update_param(self, param, grad, state):
        """Updates a parameter array and the corresponding state.

        This method can be overridden by gradient method implementations when
        the same routine can be used both for CPU and GPU arrays. If CPU and
        GPU routines should be divided, override the :meth:`update_param_cpu`
        and :meth:`update_param_gpu` methods, instead.

        Args:
            param (array): The parameter array to be updated.
            grad (array): The corresponding gradient array.
            state (dict): The corresponding state dictionary to be possibly
                updated.

        """
        if isinstance(param, cuda.ndarray):
            self.update_param_gpu(param, grad, state)
        else:
            self.update_param_cpu(param, grad, state)

    def update_param_cpu(self, param, grad, state):
        """Updates a parameter array and the corresponding state on CPU.

        Args:
            param (numpy.ndarray): The parameter array to be updated.
            grad (numpy.ndarray): The corresponding gradient array.
            state (dict): The corresponding state dictionary to be possible
                updated.

        """
        raise NotImplementedError

    def update_param_gpu(self, param, grad, state):
        """Updates a parameter array and the corresponding state on GPU.

        Args:
            param (cupy.ndarray): The parameter array to be updated.
            grad (numpy.ndarray): The corresponding gradient array.
            state (dict): The corresponding state dictionary to be possible
                updated.

        """
        raise NotImplementedError


class WeightDecay(object):

    """Optimizer hook function for weight decay.

    This hook function implements weight decay regularizer. Instead of
    implementing weight decay as a L2 regularization term, this hook function
    directly modifies the gradient arrays by adding ``decay * parameter`` to
    the corresponding gradient.

    Args:
        decay (float): Weight decay coefficient.

    """
    def __init__(self, decay):
        self.decay = decay

    def __call__(self, target):
        decay = self.decay
        for _, param in target.visitparams():
            data = param.data
            with cuda.get_device(data):
                param.grad += data.dtype.type(decay) * data


class GradientClipping(object):

    """Optimizer hook function for gradinet clipping.

    This hook function implements gradient clipping, which scales the whole
    gradient array by an appropriate scalar to fit into the given maximum L2
    norm if needed.

    Args:
        maxnorm (float): Maximum L2 norm of the resulting whole gradient array.

    """
    def __init__(self, maxnorm):
        self.maxnorm = maxnorm

    def __call__(self, target):
        # TODO(beam2d): Make it fast on GPU
        grads = []
        for _, param in target.visitparams():
            grads.append(param.grad)

        sqnorm = 0
        for grad in grads:
            grad = grad.ravel()
            sqnorm += float(grad.dot(grad))

        ratio = self.maxnorm / math.sqrt(sqnorm)
        if ratio < 1:
            for grad in grads:
                grad *= ratio
