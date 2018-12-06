# This file defines inefficient workaround implementation for
# NumPy ndarray-compatibility functions. This file should ultimately be
# emptied by implementing those functions in more efficient manner.

import chainerx


# Populates chainerx.ndarray methods in the chainerx namespace
def populate():

    def clip(self, a_min, a_max):
        """Returns an array with values limited to [``a_min``, ``a_max``].

        .. seealso:: :func:`chainerx.clip` for full documentation,
            :meth:`numpy.ndarray.clip`

        """
        return chainerx.clip(self, a_min, a_max)

    def ravel(self):
        """Returns an array flattened into one dimension.

        .. seealso:: :func:`chainerx.ravel` for full documentation,
            :meth:`numpy.ndarray.ravel`

        """
        return chainerx.ravel(self)

    chainerx.ndarray.clip = clip
    chainerx.ndarray.ravel = ravel
