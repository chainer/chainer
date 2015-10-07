import os
import weakref

from chainer import cuda
from chainer.utils import type_check
from chainer import variable


class Function(object):

    """Function on variables with backpropagation ability.

    All function implementations defined in :mod:`chainer.functions` inherit
    this class.

    The main feature of this class is keeping track of function applications as
    a backward graph. When a function is applied to :class:`Variable` objects,
    its :meth:`forward` method is called on :data:`~Variable.data` fields of
    input variables, and at the same time it chains references from output
    variables to the function and from the function to its inputs.

    Note that a function instance cannot be used multiple times in a network.
    You must create a distinct instance for each forward call.

    .. note::
       There are following notable changes at v1.4.

       - Special Function objects called *splitters* are not inserted anymore.
         In previous versions, they are required to handle graph branching
         (i.e. multiple uses of one variable). As of v1.4, the backward
         procedure itself handles branching correctly.
       - As noted above, one function instance cannot be used multiple times
         anymore, and a function instance cannot hold internal states and
         parameters. In previous versions, the function instance is copied at
         the beginning of :meth:`__call__` operator. This restriction avoids
         the copy overhead. In order to define a conventional callable object
         with states and parameters, use :class:`Link` instead.

    .. admonition:: Example

       Let ``x`` an instance of :class:`Variable` and ``f`` an instance of
       :class:`Function` taking only one argument. Then a line

       >>> y = f(x)

       computes a new variable ``y`` and creates backward references. Actually,
       backward references are set as per the following diagram::

           x <--- f <--- y

       If another application of the function occurs as

       >>> z = g(x)

       then the computational graph grows as the following new diagram::

                 |--- f <--- y
           x <---+
                 |--- g <--- z

       It means ``f`` and ``g`` are both pointing to ``x``.

    Every function implementation should provide :meth:`forward_cpu`,
    :meth:`forward_gpu`, :meth:`backward_cpu` and :meth:`backward_gpu`.
    Alternatively, one can provide :meth:`forward` and :meth:`backward` instead
    of separate methods. Backward methods have default implementations that
    just return ``None``, which indicates that the function is non-
    differentiable.

    Attributes:
        inputs: A tuple or list of input variables.
        outputs: A tuple or list of output variables.
        type_check_enable: When it is ``True``, the function checks types of
            input arguments. Set ``CHAINER_TYPE_CHECK`` environment variable
            ``0`` to disable type check, or set the variable directly in
            your own program.

    """
    type_check_enable = int(os.environ.get('CHAINER_TYPE_CHECK', '1')) != 0

    def __call__(self, *inputs):
        """Applies forward propagation with chaining backward references.

        Basic behavior is also expressed in documentation of :class:`Function`
        class.

        .. note::
           If the :data:`~Variable.data` attribute of input variables reside on
           GPU device, then, before it calls :meth:`forward` method, the
           appropriate device is selected, so in most cases implementers do
           not need to take care of device selection.

        Args:
            inputs: Tuple of input :class:`Variable` objects. All input
                variables must have same volatile flag.

        Returns:
            One
            :class:`Variable` object or a tuple of multiple
            :class:`Variable` objects.

        """
        in_data = tuple([x.data for x in inputs])
        if self.type_check_enable:
            self._check_data_type_forward(in_data)
        with cuda.get_device(*in_data):
            outputs = self.forward(in_data)
        assert type(outputs) == tuple

        volatiles = [x.volatile for x in inputs]
        if any(volatiles):
            # Do not mix multiple volatilities
            assert all(volatiles)
            ret = tuple([variable.Variable(y, volatile=True) for y in outputs])
        else:
            # Topological ordering
            self.rank = max([x.rank for x in inputs]) if inputs else 0
            for x in inputs:
                x.n_users += 1

            ret = tuple([variable.Variable(y) for y in outputs])
            # backward edges
            for y in ret:
                y.set_creator(self)
            self.inputs = inputs
            # forward edges (must be weak references)
            self.outputs = tuple([weakref.ref(y) for y in ret])

        if len(ret) == 1:
            return ret[0]
        return ret

    def __del__(self):
        inputs = getattr(self, 'inputs', None)
        if inputs is not None:
            for x in self.inputs:
                x.n_users -= 1

    @property
    def label(self):
        """Short text that represents the function.

        The default implementation returns its type name.
        Each function should override it to give more information.
        """
        return self.__class__.__name__

    def _check_data_type_forward(self, in_data):
        in_type = type_check.get_types(in_data, 'in_types', False)
        self.check_type_forward(in_type)

    def check_type_forward(self, in_types):
        """Checks types of input data before forward propagation.

        Before :meth:`forward` is called, this function is called.
        You need to validate types of input data in this function
        using :ref:`the type checking utilities <type-check-utils>`.

        Args:
            in_types (~chainer.utils.type_check.TypeInfoTuple): The type
                information of input data for :meth:`forward`.
        """
        pass

    def forward(self, inputs):
        """Applies forward propagation to input arrays.

        It delegates the procedure to :meth:`forward_cpu` or
        :meth:`forward_gpu` by default. Which it selects is determined by the
        type of input arrays.
        Implementations of :class:`Function` must implement either cpu/gpu
        methods or this method.

        Args:
            inputs: Tuple of input array(s).

        Returns:
            Tuple of output array(s).

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        if any(isinstance(x, cuda.ndarray) for x in inputs):
            return self.forward_gpu(inputs)
        else:
            return self.forward_cpu(inputs)

    def forward_cpu(self, inputs):
        """Applies forward propagation to input arrays on CPU.

        Args:
            inputs: Tuple of :class:`numpy.ndarray` object(s).

        Returns:
            tuple: Tuple of :class:`numpy.ndarray` object(s).

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        raise NotImplementedError()

    def forward_gpu(self, inputs):
        """Applies forward propagation to input arrays on GPU.

        Args:
            inputs: Tuple of :class:`cupy.ndarray` object(s).

        Returns:
            tuple: Tuple of :class:`cupy.ndarray` object(s).

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        raise NotImplementedError()

    def backward(self, inputs, grad_outputs):
        """Applies backprop to output gradient arrays.

        It delegates the procedure to :meth:`backward_cpu` or
        :meth:`backward_gpu` by default. Which it selects is determined by the
        type of input arrays and output gradient arrays. Implementations of
        :class:`Function` must implement either cpu/gpu methods or this method,
        if the function is intended to be backprop-ed.

        Args:
            inputs: Tuple of input arrays.
            grad_outputs: Tuple of output gradient arrays.

        Returns:
            tuple: Tuple of input gradient arrays. Some or all of them can be
            ``None``, if the function is not differentiable on
            inputs.

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        if any(isinstance(x, cuda.ndarray) for x in inputs + grad_outputs):
            return self.backward_gpu(inputs, grad_outputs)
        else:
            return self.backward_cpu(inputs, grad_outputs)

    def backward_cpu(self, inputs, grad_outputs):
        """Applies backprop to output gradient arrays on CPU.

        Args:
            inputs: Tuple of input :class:`numpy.ndarray` object(s).
            grad_outputs: Tuple of output gradient :class:`numpy.ndarray`
                object(s).

        Returns:
            tuple: Tuple of input gradient :class:`numpy.ndarray` object(s).
            Some or all of them can be ``None``, if the function is not
            differentiable on corresponding inputs.

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        return tuple(None for _ in inputs)

    def backward_gpu(self, inputs, grad_outputs):
        """Applies backprop to output gradient arrays on GPU.

        Args:
            inputs: Tuple of input :class:`cupy.ndarray`
                object(s).
            grad_outputs: Tuple of output gradient
                :class:`cupy.ndarray` object(s).

        Returns:
            tuple: Tuple of input gradient :class:`cupy.ndarray`
            object(s). Some or all of them can be ``None``, if the function is
            not differentiable on corresponding inputs.

        .. warning::

            Implementations of :class:`Function` must take care that the
            return value must be a tuple even if it returns only one array.

        """
        return tuple(None for _ in inputs)

    def unchain(self):
        """Purges in/out variables and this function itself from the graph.

        This method is called from :meth:`Variable.unchain_backward` method.

        """
        for y in self.outputs:
            y_ref = y()
            if y_ref is not None:
                y_ref.creator = None
        self.inputs = None
