import chainer


class FunctionHook(object):
    """Base class of hooks for Functions.

    :class:`~chainer.FunctionHook` is a callback object
    that is registered to :class:`~chainer.FunctionNode`.
    Registered function hooks are invoked before and after
    forward and backward operations of each function.

    Function hooks that derive :class:`FunctionHook` are required
    to implement four methods:
    :meth:`~chainer.FunctionHook.forward_preprocess`,
    :meth:`~chainer.FunctionHook.forward_postprocess`,
    :meth:`~chainer.FunctionHook.backward_preprocess`, and
    :meth:`~chainer.FunctionHook.backward_postprocess`.
    By default, these methods do nothing.

    Specifically, when :meth:`~chainer.FunctionNode.__call__`
    method of some function is invoked,
    :meth:`~chainer.FunctionHook.forward_preprocess`
    (resp. :meth:`~chainer.FunctionHook.forward_postprocess`)
    of all function hooks registered to this function are called before
    (resp. after) forward propagation.

    Likewise, when :meth:`~chainer.Variable.backward` of some
    :class:`~chainer.Variable` is invoked,
    :meth:`~chainer.FunctionHook.backward_preprocess`
    (resp. :meth:`~chainer.FunctionHook.backward_postprocess`)
    of all function hooks registered to the function which holds this variable
    as a gradient are called before (resp. after) backward propagation.

    There are two ways to register :class:`~chainer.FunctionHook`
    objects to :class:`~chainer.FunctionNode` objects.

    First one is to use ``with`` statement. Function hooks hooked
    in this way are registered to all functions within ``with`` statement
    and are unregistered at the end of ``with`` statement.

    .. admonition:: Example

        The following code is a simple example in which
        we measure the elapsed time of a part of forward propagation procedure
        with :class:`~chainer.function_hooks.TimerHook`, which is a subclass of
        :class:`~chainer.FunctionHook`.

        >>> from chainer import function_hooks
        >>> class Model(chainer.Chain):
        ...   def __init__(self):
        ...     super(Model, self).__init__()
        ...     with self.init_scope():
        ...       self.l = L.Linear(10, 10)
        ...   def __call__(self, x1):
        ...     return F.exp(self.l(x1))
        >>> model1 = Model()
        >>> model2 = Model()
        >>> x = chainer.Variable(np.zeros((1, 10), 'f'))
        >>> with chainer.function_hooks.TimerHook() as m:
        ...   _ = model1(x)
        ...   y = model2(x)
        ...   print("Total time : " + str(m.total_time()))
        ...   model3 = Model()
        ...   z = model3(y) # doctest:+ELLIPSIS
        Total time : ...

        In this example, we measure the elapsed times for each forward
        propagation of all functions in ``model1`` and ``model2``
        (specifically, :class:`~chainer.functions.LinearFunction` and
        :class:`~chainer.functions.Exp` of ``model1`` and ``model2``).
        Note that ``model3`` is not a target of measurement
        as :class:`~chainer.function_hooks.TimerHook` is unregistered
        before forward propagation of ``model3``.

    .. note::

       Chainer stores the dictionary of registered function hooks
       as a thread local object. So, function hooks registered
       are different depending on threads.

    The other one is to register directly to
    :class:`~chainer.FunctionNode` object with
    :meth:`~chainer.Function.add_hook` method.
    Function hooks registered in this way can be removed by
    :meth:`~chainer.Function.delete_hook` method.
    Contrary to former registration method, function hooks are registered
    only to the function which :meth:`~chainer.FunctionNode.add_hook`
    is called.

    Args:
        name(str): Name of this function hook.
    """

    name = 'FunctionHook'

    def __enter__(self):
        function_hooks = chainer.get_function_hooks()
        if self.name in function_hooks:
            raise KeyError('hook %s already exists' % self.name)

        function_hooks[self.name] = self
        self.added()
        return self

    def __exit__(self, *_):
        chainer.get_function_hooks()[self.name].deleted()
        del chainer.get_function_hooks()[self.name]

    def added(self, function=None):
        """Callback function invoked when a function hook is added

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is added.
        """
        pass

    def deleted(self, function=None):
        """Callback function invoked when a function hook is deleted

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is deleted.
        """
        pass

    # forward
    def forward_preprocess(self, function, in_data):
        """Callback function invoked before forward propagation.

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is registered.
            in_data(tuple of numpy.ndarray or tuple of cupy.ndarray):
               Input data of forward propagation.
        """
        pass

    def forward_postprocess(self, function, in_data):
        """Callback function invoked after forward propagation.

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is registered.
            in_data(tuple of numpy.ndarray or tuple of cupy.ndarray):
                Input data of forward propagation.
        """
        pass

    # backward
    def backward_preprocess(self, function, in_data, out_grad):
        """Callback function invoked before backward propagation.

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is registered.
            in_data(tuple of numpy.ndarray or tuple of cupy.ndarray):
                Input data of forward propagation.
            out_grad(tuple of numpy.ndarray or tuple of cupy.ndarray):
                Gradient data of backward propagation.
        """
        pass

    def backward_postprocess(self, function, in_data, out_grad):
        """Callback function invoked after backward propagation.

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is registered.
            in_data(tuple of numpy.ndarray or tuple of cupy.ndarray):
                Input of forward propagation.
            out_grad(tuple of numpy.ndarray or tuple of cupy.ndarray):
                Gradient data of backward propagation.
        """
        pass
