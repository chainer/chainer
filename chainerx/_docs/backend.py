import chainerx
from chainerx import _docs


def _set_docs_backend():
    Backend = chainerx.Backend

    _docs.set_doc(
        Backend,
        """Pluggable entity that abstracts various computing platforms.

A backend holds one or more :class:`~chainerx.Device`\\ s, each of which
represents a physical computing unit.
""")

    _docs.set_doc(
        Backend.name,
        """Backend name.

Returns:
    str: Backend name.
""")

    _docs.set_doc(
        Backend.context,
        """Context to which this backend belongs.

Returns:
    ~chainerx.Context: Context object.

""")

    _docs.set_doc(
        Backend.get_device,
        """get_device(index)
Returns a device specified by the given index.

Args:
    index (int): Device index.

Returns:
    ~chainerx.Device: Device object.
""")

    _docs.set_doc(
        Backend.get_device_count,
        """get_device_count()
Returns the number of devices available in this backend.

Returns:
    int: Number of devices.
""")


def set_docs():
    _set_docs_backend()

    _docs.set_doc(
        chainerx.get_backend,
        """get_backend(backend_name)
Returns a backend specified by the name.

Args:
    backend_name (str): Backend name.

Returns:
    ~chainerx.Backend: Backend object.
""")
