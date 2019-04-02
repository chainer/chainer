class AbstractSerializer(object):

    """Abstract base class of all serializers and deserializers."""

    def __getitem__(self, key):
        """Gets a child serializer.

        This operator creates a _child_ serializer represented by the given
        key.

        Args:
            key (str): Name of the child serializer.

        """
        raise NotImplementedError

    def __call__(self, key, value):
        """Serializes or deserializes a value by given name.

        This operator saves or loads a value by given name.

        If this is a serializer, then the value is simply saved at the key.
        Note that some type information might be missed depending on the
        implementation (and the target file format).

        If this is a deserializer, then the value is loaded by the key. The
        deserialization differently works on scalars and arrays. For scalars,
        the ``value`` argument is used just for determining the type of
        restored value to be converted, and the converted value is returned.
        For arrays, the restored elements are directly copied into the
        ``value`` argument. String values are treated like scalars.

        .. note::
           Serializers and deserializers are required to
           correctly handle the ``None`` value. When ``value`` is ``None``,
           serializers save it in format-dependent ways, and deserializers
           just return the loaded value. When the saved ``None`` value is
           loaded by a deserializer, it should quietly return the ``None``
           value without modifying the ``value`` object.

        Args:
            key (str): Name of the serialization entry.
            value (scalar, numpy.ndarray, cupy.ndarray, None, or str):
                Object to be (de)serialized.
                ``None`` is only supported by deserializers.

        Returns:
            Serialized or deserialized value.

        """
        raise NotImplementedError


class Serializer(AbstractSerializer):

    """Base class of all serializers."""

    def save(self, obj):
        """Saves an object by this serializer.

        This is equivalent to ``obj.serialize(self)``.

        Args:
            obj: Target object to be serialized.

        """
        obj.serialize(self)


class Deserializer(AbstractSerializer):

    """Base class of all deserializers."""

    def load(self, obj):
        """Loads an object from this deserializer.

        This is equivalent to ``obj.serialize(self)``.

        Args:
            obj: Target object to be serialized.

        """
        obj.serialize(self)
