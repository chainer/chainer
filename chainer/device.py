import contextlib

from chainer import utils


@contextlib.contextmanager
def _dummy_context():
    yield


# TODO(niboshi): Write more detailed description about interface/usage.
class Device(object):
    """Device object.
    """

    @property
    def xp(self):
        """Array module corresponding to the device."""
        raise NotImplementedError(
            'Device implementation must override this property.')

    def __enter__(self):
        raise RuntimeError(
            'Device class does not support runtime context using `with` '
            'statement. Use chainer.using_device instead.')

    def __exit__(self, exc_type, exc_value, traceback):
        # Definition of __exit__ is needed to raise a custom error on
        # __enter__.
        pass

    def __eq__(self, other):
        raise NotImplementedError(
            'Device implementation must override this method.')

    def create_context(self):
        # Returns an object that implements __enter__ and __exit__.
        return _dummy_context()

    def send(self, arrays):
        """Transfers given arrays to the device.

        Args:
            arrays: Array or arrays of NumPy, CuPy, or ChainerX.

        Returns:
            Transferred arrays.

        """
        return utils.array._convert_arrays(arrays, self.send_array)
