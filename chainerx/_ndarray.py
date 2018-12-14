# This file implements chainerx.ndarray methods that can be defined only in
# Python.

import chainerx


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
