import chainer


class FunctionHook(object):
    """Base class of hooks for Functions.

    :class:`~chainer.FunctionHook` is a callback object
    that is registered to :class:`~chainer.FunctionNode`.
    Registered function hooks are invoked before and after
    forward and backward operations of each function.

    Function hooks that derive from :class:`FunctionHook` may override the
    following  methods:

    * :meth:`~chainer.FunctionHook.added`
    * :meth:`~chainer.FunctionHook.deleted`
    * :meth:`~chainer.FunctionHook.forward_preprocess`
    * :meth:`~chainer.FunctionHook.forward_postprocess`
    * :meth:`~chainer.FunctionHook.backward_preprocess`
    * :meth:`~chainer.FunctionHook.backward_postprocess`

    By default, these methods do nothing.

    Specifically, when the :meth:`~chainer.FunctionNode.__call__`
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

    :meth:`~chainer.FunctionHook.added` and
    :meth:`~chainer.FunctionHook.deleted` are called when the hook is
    registered or unregistered, respectively.

    There are two ways to register :class:`~chainer.FunctionHook`
    objects to :class:`~chainer.FunctionNode` objects.

    The first one is to use ``with`` statement. Function hooks hooked
    in this way are registered to all functions within ``with`` statement
    and are unregistered at the end of ``with`` statement.

    .. admonition:: Example

        The following code is a simple example in which
        we measure the elapsed time of a part of forward propagation procedure
        with :class:`~chainer.function_hooks.TimerHook`, which is a subclass of
        :class:`~chainer.FunctionHook`.

        >>> class Model(chainer.Chain):
        ...   def __init__(self):
        ...     super(Model, self).__init__()
        ...     with self.init_scope():
        ...       self.l = L.Linear(10, 10)
        ...   def __call__(self, x1):
        ...     return F.exp(self.l(x1))
        >>> model1 = Model()
        >>> model2 = Model()
        >>> x = chainer.Variable(np.zeros((1, 10), np.float32))
        >>> with chainer.function_hooks.TimerHook() as m:
        ...   _ = model1(x)
        ...   y = model2(x)
        >>> model3 = Model()
        >>> z = model3(y)
        >>> print('Total time : {}'.format(m.total_time()))
        ... # doctest:+ELLIPSIS
        Total time : ...

        In this example, we measure the elapsed times for each forward
        propagation of all functions in ``model1`` and ``model2``.
        Note that ``model3`` is not a target of measurement
        as :class:`~chainer.function_hooks.TimerHook` is unregistered
        before forward propagation of ``model3``.

    .. note::

       Chainer stores the dictionary of registered function hooks
       as a thread local object. So, function hooks registered
       are different depending on threads.

    The other one is to register it directly to
    a :class:`~chainer.FunctionNode` object by calling its
    :meth:`~chainer.FunctionNode.add_hook` method.
    Function hooks registered in this way can be removed by
    :meth:`~chainer.FunctionNode.delete_hook` method.
    Contrary to the former registration method, function hooks are registered
    only to the function whose :meth:`~chainer.FunctionNode.add_hook`
    method is called.

    If the hook is registered globally using ``with`` statement, ``None`` is
    passed as the ``function`` argument of :meth:`~chainer.FunctionHook.added`
    and :meth:`~chainer.FunctionHook.deleted`.

    If the hook is registered in a specific function using
    :meth:`~chainer.FunctionNode.add_hook`, the :class:`~chainer.FunctionNode`
    instance is passed as the ``function`` argument of
    :meth:`~chainer.FunctionHook.added` and
    :meth:`~chainer.FunctionHook.deleted`.

    Args:
        name(str): Name of this function hook.
    """

    name = 'FunctionHook'

    def __enter__(self):
        function_hooks = chainer.get_function_hooks()
        if self.name in function_hooks:
            raise KeyError('hook %s already exists' % self.name)

        function_hooks[self.name] = self
        self.added(None)
        return self

    def __exit__(self, *_):
        chainer.get_function_hooks()[self.name].deleted(None)
        del chainer.get_function_hooks()[self.name]

    def added(self, function):
        """Callback function invoked when the function hook is registered

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is added. ``None`` if the function hook is
                registered globally.
        """
        pass

    def deleted(self, function):
        """Callback function invoked when the function hook is unregistered

        Args:
            function(~chainer.FunctionNode): Function object from which
                the function hook is deleted. ``None`` if the function hook
                was registered globally.
        """
        pass

    # forward
    def forward_preprocess(self, function, in_data):
        """Callback function invoked before forward propagation.

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is registered.
            in_data(tuple of :ref:`ndarray`):
               Input data of forward propagation.
        """
        pass

    def forward_postprocess(self, function, in_data):
        """Callback function invoked after forward propagation.

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is registered.
            in_data(tuple of :ref:`ndarray`):
                Input data of forward propagation.
        """
        pass

    # backward
    def backward_preprocess(self, function, in_data, out_grad):
        """Callback function invoked before backward propagation.

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is registered.
            in_data(tuple of :ref:`ndarray`):
                Input data of forward propagation.
            out_grad(tuple of :ref:`ndarray`):
                Gradient data of backward propagation.
        """
        pass

    def backward_postprocess(self, function, in_data, out_grad):
        """Callback function invoked after backward propagation.

        Args:
            function(~chainer.FunctionNode): Function object to which
                the function hook is registered.
            in_data(tuple of :ref:`ndarray`):
                Input of forward propagation.
            out_grad(tuple of :ref:`ndarray`):
                Gradient data of backward propagation.
        """
        pass
