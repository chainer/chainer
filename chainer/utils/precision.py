import functools

import numpy


def _fp16_mixed_precision_helper(fn):
    """Decorator to perform computation in FP32 for FP16 inputs/outputs

    Decorator to perform forward computation in FP32 for FP16 inputs,
    returning outputs casted back to FP16. Do nothing for FP32 and FP64
    inputs.
    """
    @functools.wraps(fn)
    def wrapper(self, in_data):
        flag = all([x.dtype.kind != 'f' or x.dtype == numpy.float16
                    for x in in_data])

        in_data1 = []
        for x in in_data:
            if x.dtype == numpy.float16:
                in_data1.append(x.astype(numpy.float32))
            else:
                in_data1.append(x)
        in_data1 = tuple(in_data1)

        out_data = fn(self, in_data1)

        if flag:
            out_data1 = []
            for y in out_data:
                if y is not None and y.dtype == numpy.float32:
                    out_data1.append(y.astype(numpy.float16))
                else:
                    out_data1.append(y)
            out_data = tuple(out_data1)

        return out_data
    return wrapper
