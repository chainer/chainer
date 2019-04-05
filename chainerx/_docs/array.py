import chainerx
from chainerx import _docs


def set_docs():
    ndarray = chainerx.ndarray

    _docs.set_doc(
        ndarray,
        """ndarray(shape, dtype, device=None)
Multi-dimensional array, the central data structure of ChainerX.

This class, along with other APIs in the :mod:`chainerx` module, provides a
subset of NumPy APIs. This class works similar to :class:`numpy.ndarray`,
except for some differences including the following noticeable points:

- :class:`chainerx.ndarray` has a :attr:`device` attribute. It indicates on
  which device the array is allocated.
- :class:`chainerx.ndarray` supports :ref:`Define-by-Run <define_by_run>`
  backpropagation. Once you call :meth:`require_grad`, the array starts
  recording the operations applied to it recursively. Gradient of the result
  with respect to the original array can be computed then with the
  :meth:`backward` method or the :func:`chainerx.backward` function.

Args:
    shape (tuple of ints): Shape of the new array.
    dtype: Data type.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

.. seealso:: :class:`numpy.ndarray`
""")

    _docs.set_doc(
        ndarray.data_ptr,
        """int: Address of the underlying memory allocation.

The meaning of the address is device-dependent.
""")

    _docs.set_doc(
        ndarray.data_size,
        'int: Total size of the underlying memory allocation.')

    _docs.set_doc(
        ndarray.device, '~chainerx.Device: Device on which the data exists.')

    _docs.set_doc(ndarray.dtype, 'Data type of the array.')

    # TODO(beam2d): Write about backprop id.
    _docs.set_doc(
        ndarray.grad,
        """~chainerx.ndarray: Gradient held by the array.

It is ``None`` if the gradient is not available.
Setter of this property overwrites the gradient.
""")

    _docs.set_doc(
        ndarray.is_contiguous,
        'bool: ``True`` iff the array is stored in the C-contiguous order.')

    _docs.set_doc(ndarray.itemsize, 'int: Size of each element in bytes.')

    _docs.set_doc(
        ndarray.nbytes,
        """int: Total size of all elements in bytes.

It does not count skips between elements.""")

    _docs.set_doc(ndarray.ndim, 'int: Number of dimensions.')

    _docs.set_doc(
        ndarray.offset,
        'int: Offset of the first element from the memory allocation in bytes.'
    )

    _docs.set_doc(
        ndarray.shape,
        """tuple of int: Lengths of axes.

.. note::
    Currently, this property does not support setter.""")

    _docs.set_doc(ndarray.size, 'int: Number of elements in the array.')

    _docs.set_doc(ndarray.strides, 'tuple of int: Strides of axes in bytes.')

    _docs.set_doc(
        ndarray.T,
        """~chainerx.ndarray: Shape-reversed view of the array.

New array is created at every access to this property.
``x.T`` is just a shorthand of ``x.transpose()``.
""")

    _docs.set_doc(
        ndarray.__getitem__,
        """___getitem__(self, key)
Returns self[key].

.. note::
    Currently, only basic indexing is supported not advanced indexing.
""")

    def unary_op(name, s):
        _docs.set_doc(getattr(ndarray, name), '{}()\n{}'.format(name, s))

    unary_op('__bool__', 'Casts a size-one array into a :class:`bool` value.')
    unary_op('__float__',
             'Casts a size-one array into a :class:`float` value.')
    unary_op('__int__', 'Casts a size-one array into :class:`int` value.')
    unary_op('__len__', 'Returns the length of the first axis.')
    unary_op('__neg__', 'Computes ``-x`` elementwise.')

    def binary_op(name, s):
        _docs.set_doc(getattr(ndarray, name), '{}(other)\n{}'.format(name, s))

    binary_op('__eq__', 'Computes ``x == y`` elementwise.')
    binary_op('__ne__', 'Computes ``x != y`` elementwise.')
    binary_op('__lt__', 'Computes ``x < y`` elementwise.')
    binary_op('__le__', 'Computes ``x <= y`` elementwise.')
    binary_op('__ge__', 'Computes ``x >= y`` elementwise.')
    binary_op('__gt__', 'Computes ``x > y`` elementwise.')

    binary_op('__iadd__', 'Computes ``x += y`` elementwise.')
    binary_op('__isub__', 'Computes ``x -= y`` elementwise.')
    binary_op('__imul__', 'Computes ``x *= y`` elementwise.')
    binary_op('__itruediv__', 'Computes ``x /= y`` elementwise.')

    binary_op('__add__', 'Computes ``x + y`` elementwise.')
    binary_op('__sub__', 'Computes ``x - y`` elementwise.')
    binary_op('__mul__', 'Computes ``x * y`` elementwise.')
    binary_op('__truediv__', 'Computes ``x / y`` elementwise.')

    binary_op('__radd__', 'Computes ``y + x`` elementwise.')
    binary_op('__rsub__', 'Computes ``y - x`` elementwise.')
    binary_op('__rmul__', 'Computes ``y * x`` elementwise.')

    # TODO(beam2d): Write about as_grad_stopped(backprop_ids, copy) overload.
    _docs.set_doc(
        ndarray.as_grad_stopped,
        """as_grad_stopped(copy=False)
Creates a view or a copy of the array that stops gradient propagation.

This method behaves similar to :meth:`view` and :meth:`copy`, except that
the gradient is not propagated through this operation (internally, this
method creates a copy or view of the array without connecting the computational
graph for backprop).

Args:
    copy (bool): If ``True``, it copies the array. Otherwise, it returns a view
        of the original array.

Returns:
    ~chainerx.ndarray:
        A view or a copy of the array without propagating the  gradient on
        backprop.
""")

    _docs.set_doc(
        ndarray.argmax,
        """argmax(axis=None)
Returns the indices of the maximum elements along a given axis.

See :func:`chainerx.argmax` for the full documentation.
""")

    _docs.set_doc(
        ndarray.astype,
        """astype(dtype, copy=True)
Casts each element to the specified data type.

Args:
    dtype: Data type of the new array.
    copy (bool): If ``True``, this method always copies the data. Otherwise,
        it creates a view of the array if possible.

Returns:
    ~chainerx.ndarray: An array with the specified dtype.
""")

    _docs.set_doc(
        ndarray.backward,
        """backward(backprop_id=None, enable_double_backprop=False)
Performs backpropagation starting from this array.

This method is equivalent to ``chainerx.backward([self], *args)``.
See :func:`chainerx.backward` for the full documentation.
""")

    # TODO(beam2d): Write about backprop id.
    _docs.set_doc(
        ndarray.cleargrad,
        """cleargrad()
Clears the gradient held by this array.
""")

    _docs.set_doc(
        ndarray.copy,
        """copy()
Creates an array and copies all the elements to it.

The copied array is allocated on the same device as ``self``.

.. seealso:: :func:`chainerx.copy`
""")

    _docs.set_doc(
        ndarray.dot,
        """dot(b)
Returns the dot product with a given array.

See :func:`chainerx.dot` for the full documentation.
""")

    _docs.set_doc(
        ndarray.fill,
        """fill(value)
Fills the array with a scalar value in place.

Args:
    value: Scalar value with which the array will be filled.
""")

    # TODO(beam2d): Write about backprop_id argument.
    _docs.set_doc(
        ndarray.get_grad,
        """get_grad()
Returns the gradient held by the array.

If the gradient is not available, it returns ``None``.
""")

    # TODO(beam2d): Write about backprop_id argument.
    _docs.set_doc(
        ndarray.is_backprop_required,
        """is_backprop_required()
Returns ``True`` if gradient propagates through this array on backprop.

See the note on :meth:`require_grad` for details.
""")

    # TODO(beam2d): Write about backprop_id argument.
    _docs.set_doc(
        ndarray.is_grad_required,
        """is_grad_required()
Returns ``True`` if the gradient will be set after backprop.

See the note on :meth:`require_grad` for details.
""")

    _docs.set_doc(
        ndarray.item,
        """item()
Copies an element of an array to a standard Python scalar and returns it.

Returns:
    z:
        A copy of the specified element of the array as a suitable Python
        scalar.

.. seealso:: :func:`numpy.item`
""")

    _docs.set_doc(
        ndarray.max,
        """max(axis=None, keepdims=False)
Returns the maximum along a given axis.

See :func:`chainerx.amax` for the full documentation.
""")

    # TODO(beam2d): Write about backprop_id argument.
    _docs.set_doc(
        ndarray.require_grad,
        """require_grad()
Declares that a gradient for this array will be made available after backprop.

Once calling this method, any operations applied to this array are recorded for
later backprop. After backprop, the :attr:`grad` attribute holds the gradient
array.

.. note::
    ChainerX distinguishes *gradient requirements* and *backprop requirements*
    strictly. They are strongly related, but different concepts as follows.

    - *Gradient requirement* indicates that the gradient array should be made
      available after backprop. This attribute **is not propagated** through
      any operations. It implicates the backprop requirement.
    - *Backprop requirement* indicates that the gradient should be propagated
      through the array during backprop. This attribute **is propagated**
      through differentiable operations.

    :meth:`require_grad` sets the gradient requirement flag. If you need to
    extract the gradient after backprop, you have to call :meth:`require_grad`
    on the array even if the array is an intermediate result of differentiable
    computations.

Returns:
    ~chainerx.ndarray: ``self``
""")

    _docs.set_doc(
        ndarray.reshape,
        """reshape(newshape)
Creates an array with a new shape and the same data.

See :func:`chainerx.reshape` for the full documentation.
""")

    _docs.set_doc(
        ndarray.set_grad,
        """set_grad(grad)
Sets a gradient to the array.

This method overwrites the gradient with a given array.

Args:
    grad (~chainerx.ndarray): New gradient array.
""")

    _docs.set_doc(
        ndarray.squeeze,
        """squeeze(axis=None)
Removes size-one axes from an array.

See :func:`chainerx.squeeze` for the full documentation.
""")

    _docs.set_doc(
        ndarray.sum,
        """sum(axis=None, keepdims=False)
Returns the sum of an array along given axes.

See :func:`chainerx.sum` for the full documentation.
""")

    _docs.set_doc(
        ndarray.take,
        """take(indices, axis)
Takes elements from the array along an axis.

See :func:`chainerx.take` for the full documentation.
""")

    _docs.set_doc(
        ndarray.to_device,
        """to_device(device, index=None)
Transfers the array to the specified device.

Args:
    device (~chainerx.Device or str): Device to which the array is transferred,
        or a backend name. If it is a backend name, ``index`` should also be
        specified.
    index (int): Index of the device for the backend specified by ``device``.

Returns:
    ~chainerx.ndarray:
        An array on the target device.
        If the original array is already on the device, it is a view of that.
        Otherwise, it is a copy of the array on the target device.
""")

    _docs.set_doc(
        ndarray.transpose,
        """transpose(axes=None)
Creates a view of an array with permutated axes.

See :func:`chainerx.transpose` for the full documentation.
""")

    _docs.set_doc(
        ndarray.view,
        """view()
Returns a view of the array.

The returned array shares the underlying buffer, though it has a different
identity as a Python object.
""")
