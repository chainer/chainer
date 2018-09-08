import numpy
import functools


def mixed_precision(fn):
    """Decorator to perform forward computation in FP32 for FP16 inputs,
       returning outputs casted back to FP16. Do nothing for FP32 and FP64
       inputs.
    """
    @functools.wraps(fn)
    def wrapper(self, in_data):
        for x in in_data:
            assert x.dtype.kind == 'f'  # should check in check_type_forward
        mask = tuple(x.dtype == numpy.float16 for x in in_data)
        in_data = tuple(x.astype(numpy.float32) if m else x
                        for x, m in zip(in_data, mask))
        out_data = fn(self, in_data)
        out_data = tuple(y.astype(numpy.float16) if m else y
                         for y, m in zip(out_data, mask))
        return out_data
    return wrapper
