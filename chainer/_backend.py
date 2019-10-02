def _convert_arrays(array, func):
    # Converts array or arrays
    if isinstance(array, (list, tuple)):
        # The same object encountered multiple times in the container is
        # converted into the same object.
        d = {}
        ret = []
        for arr in array:
            if arr is None:
                ret.append(None)
            else:
                arr2 = d.get(id(arr))
                if arr2 is None:
                    arr2 = func(arr)
                    d[id(arr)] = arr2
                ret.append(arr2)
        return type(array)(ret)
    else:
        return func(array)


class _DummyContext(object):
    def __enter__(self):
        pass

    def __exit__(self, typ, value, traceback):
        pass


_dummy_context = _DummyContext()


# TODO(niboshi): Write more detailed description about interface/usage.
class Device(object):
    """A base class of unified devices.

    Chainer has the following concrete implementations:

    - :class:`chainer.backend.CpuDevice`
    - :class:`chainer.backend.GpuDevice`
    - :class:`chainer.backend.Intel64Device`
    - :class:`chainer.backend.ChainerxDevice`
    """

    @property
    def xp(self):
        """Array module corresponding to the device."""
        raise NotImplementedError(
            'Device implementation must override this property.')

    @property
    def name(self):
        """A unique name of the device."""
        raise NotImplementedError(
            'Device implementation must override this property.')

    @property
    def supported_array_types(self):
        """Array types supported by the device.

        Returns:
            tuple of array types which the device's module functions can
            handle.
        """
        raise NotImplementedError(
            'Device implementation must override this property.')

    def __str__(self):
        return self.name

    def __enter__(self):
        """A dummy definition that simply raises RuntimeError.

        :meth:`chainer.using_device` should be used instead.
        """
        raise RuntimeError(
            'Device class does not support runtime context using `with` '
            'statement. Use chainer.using_device instead.')

    def __exit__(self, exc_type, exc_value, traceback):
        """A dummy definition that should never be called."""
        # Definition of __exit__ is needed to raise a custom error on
        # __enter__.
        pass

    def __eq__(self, other):
        raise NotImplementedError(
            'Device implementation must override this method.')

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.name)

    def create_context(self):
        """Returns a context manager in which the device is made current.

        .. seealso::
            :meth:`chainer.using_device` calls this method internally.
        """
        return _dummy_context

    def send(self, arrays):
        """Transfers given arrays to the device.

        Args:
            arrays: Array or arrays of NumPy, CuPy, or ChainerX.

        Returns:
            Transferred arrays.

        """
        return _convert_arrays(arrays, self.send_array)

    def use(self):
        """Makes the device current in the current thread.
         """
        pass

    def is_array_supported(self, array):
        """Returns if the specified array is compatible with the device.
        Args:
            array (:ref:`ndarray`): An array to be checked
        Returns:
            ``True`` if the array is compatible with the device. Otherwise
            ``False`` is returned.
        """
        raise NotImplementedError(
            'Device implementation must override this method.')
