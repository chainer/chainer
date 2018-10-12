# This file defines inefficient workaround implementation for
# NumPy-compatibility functions. This file should ultimately be emptied by
# implementing those functions in more efficient manner.

import chainerx


def populate():
    # Populates workaround functions in the chainerx namespace
    ndarray = chainerx.ndarray

    def clip(self, a_min, a_max):
        return -chainerx.maximum(-chainerx.maximum(self, a_min), -a_max)

    def ravel(self):
        return self.reshape((self.size,))

    ndarray.clip = clip
    ndarray.ravel = ravel
