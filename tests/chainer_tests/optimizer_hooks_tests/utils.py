import numpy

import chainer


class ParametersLink(chainer.Link):
    '''Link with specific parameters.'''

    def __init__(self, params):
        super(ParametersLink, self).__init__()
        with self.init_scope():
            for i, p in enumerate(params):
                setattr(self, 'p{}'.format(i), p)

    @staticmethod
    def from_param_props(shapes, dtypes=numpy.float32):
        # Creates a ParameterLink from the given parameter properties.
        assert isinstance(shapes, (tuple, list))
        assert all(isinstance(s, tuple) for s in shapes)

        n_params = len(shapes)

        if not isinstance(dtypes, (tuple, list)):
            dtypes = (dtypes,) * n_params

        arrs = [
            numpy.random.uniform(-3, 3, shape).astype(dtype)
            for shape, dtype in zip(shapes, dtypes)]
        grads = [
            numpy.random.uniform(-3, 3, shape).astype(dtype)
            for shape, dtype in zip(shapes, dtypes)]

        params = []
        for arr, grad in zip(arrs, grads):
            param = chainer.Parameter(arr)
            param.grad = grad
            params.append(param)

        return ParametersLink(params)
