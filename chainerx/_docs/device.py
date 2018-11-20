import chainerx
from chainerx import _docs


def _set_docs_device():
    Device = chainerx.Device

    _docs.set_doc(
        Device,
        """Represents a physical computing unit.
""")

    _docs.set_doc(
        Device.synchronize,
        """Synchronizes the device.
""")

    _docs.set_doc(
        Device.name,
        """Device name.

It is the backend name and the device index concatenated with a colon, e.g.
``native:0``.

Returns:
    str: Device name.
""")

    _docs.set_doc(
        Device.backend,
        """Backend to which this device belongs.

Returns:
    ~chainerx.Backend: Backend object.
""")

    _docs.set_doc(
        Device.context,
        """Context to which this device belongs.

Returns:
    ~chainerx.Context: Context object.
""")

    _docs.set_doc(
        Device.index,
        """Index of this device.

Returns:
    int: Index of this device.
""")


def set_docs():
    _set_docs_device()

    _docs.set_doc(
        chainerx.get_device,
        """get_device(*device)
Returns a device specified by the arguments.

If the argument is a single :class:`~chainerx.Device` instance, it's simply
returned.

Otherwise, there are three ways to specify a device:


.. testcode::

    # Specify a backend name and a device index separately.
    chainerx.get_device('native', 0)

    # Specify a backend name and a device index in a single string.
    chainerx.get_device('native:0')

    # Specify only a backend name. In this case device index 0 is chosen.
    chainerx.get_device('native')

Returns:
    ~chainerx.Device: Device object.
""")

    _docs.set_doc(
        chainerx.get_default_device,
        """get_default_device()
Returns the default device associated with the current thread.

Returns:
    ~chainerx.Device: The default device.

.. seealso::
    * :func:`chainerx.set_default_device`
    * :func:`chainerx.using_device`
""")

    _docs.set_doc(
        chainerx.set_default_device,
        """set_default_device(device)
Sets the given device as the default device of the current thread.

Args:
    device (~chainerx.Device or str): Device object or device name to set as
        the default device.

.. seealso::
    * :func:`chainerx.get_default_device`
    * :func:`chainerx.using_device`
""")

    _docs.set_doc(
        chainerx.using_device,
        """using_device(device)
Creates a context manager to temporarily set the default device.

Args:
    device (~chainerx.Device or str): Device object or device name to set as
        the default device during the context. See :data:`chainerx.Device.name`
        for the specification of device names.

.. seealso::
    * :func:`chainerx.get_default_device`
    * :func:`chainerx.set_default_device`
""")
