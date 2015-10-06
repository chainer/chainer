import numpy
import warnings

from chainer.functions.math import identity
from chainer import link
from chainer import variable


class Parameter(link.Link):

    """Link that just holds a parameter and returns it.

    .. deprecated:: v1.4
       The parameters are stored as variables as of v1.4. Use them directly
       instead.

    Args:
        array: Initial parameter array.

    """
    def __init__(self, array):
        super(Parameter, self).__init__()
        self.params['W'] = variable.Variable(array)

    def __call__(self, volatile=None):
        """Returns the parameter variable.

        Args:
            volatile (bool): If specified, the volatility of the output
                variable is set to this value.

        Returns:
            ~chainer.Variable: A copy of the parameter variable.

        """
        # This is a bit tricky code. The first identity avoids modification of
        # the volatility of the internal parameter that might be referenced via
        # ``self.volatile``. The second identity avoids modification of the
        # gradient on backward if the volatility is True.
        if volatile is not None:
            W = identity.identity(self.params['W'])
            W.volatile = volatile
        else:
            W = self.params['W']  # Use the current volatility of this link
        return identity.identity(W)
