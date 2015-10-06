class Serializer(object):

    """Interface of serialization in Chainer.

    All (de)serializers in Chainer implement this interface.

    Serializer saves or loads arrays and scalars with hierarchical structure.
    One serializer object represents a node in the hierarchy. The *child* node
    can be extracted as a serializer by the :meth:`__getitem__` operator.

    Serializers support CuPy arrays as well as NumPy arrays. The saved data is
    device-agnostic, so users can save CuPy arrays and load them to NumPy
    arrays, and vice versa.

    In order to support the serialization protocol, the target object must
    implement a ``serialize`` method which takes one serializer as an argument.
    Currently, :class:`Link` objects and :class:`Optimizer` objects support the
    serialization protocol.

    Concrete serialization methods are implemented in the :mod:`serializers`
    subpackage.

    """
    @property
    def reader(self):
        """True if this is an input serializer (i.e. deserializer)."""
        return not self.writer

    @property
    def writer(self):
        """True if this is an output serializer."""
        raise NotImplementedError

    def __getitem__(self, key):
        """Creates a child node of the data hierarchy.

        Args:
            key (str): Relative name of the child node.

        Returns:
            Serializer: Serializer object representing the child node.

        """
        raise NotImplementedError

    def __call__(self, key, value):
        """Load/save the value from/to the key.

        Args:
            key (str): Relative name of the data array to be loaded/saved.
            value (scalar or array): Value to be loaded/saved. On saving, the
                value is just saved. On loading, if the value is an array, then
                it is overwritten by the loaded value. If the value is a
                scalar, then a value is loaded in the same type as the given
                value, and the loaded value is returned.

        Returns:
            Loaded/saved value. It can be used to load scalar value to a
            variable of the caller.

        """
        raise NotImplementedError
