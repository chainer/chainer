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

    chainerx.ndarray.clip = clip
