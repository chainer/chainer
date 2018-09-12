import numpy
import functools


def mixed_precision(fn):
    """Decorator to perform forward computation in FP32 for FP16 inputs,
       returning outputs casted back to FP16. Do nothing for FP32 and FP64
       inputs.
    """
    @functools.wraps(fn)
    def wrapper(self, in_data):
        flag = any([x.dtype == numpy.float16 for x in in_data])
        in_data = tuple([
            x.astype(numpy.float32) if x.dtype == numpy.float16 else x
            for x in in_data])
        out_data = fn(self, in_data)
        if flag:
            out_data = tuple([
                y.astype(numpy.float16) if y.dtype == numpy.float32 else y
                for y in out_data])
        return out_data
    return wrapper
