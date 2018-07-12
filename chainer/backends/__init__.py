import numpy

from chainer.backends import cuda


def copyto(dst, src):
    """Copies the elements of an ndarray to those of another one.

    This function can copy the CPU/GPU arrays to the destination arrays on
    another device.

    Args:
        dst (numpy.ndarray or cupy.ndarray): Destination array.
        src (numpy.ndarray or cupy.ndarray): Source array.

    """
    if isinstance(dst, numpy.ndarray):
        numpy.copyto(dst, cuda.to_cpu(src))
    elif isinstance(dst, cuda.ndarray):
        if isinstance(src, numpy.ndarray):
            if dst.flags.c_contiguous or dst.flags.f_contiguous:
                dst.set(src)
            else:
                cuda.cupy.copyto(dst, cuda.to_gpu(src, device=dst.device))
        elif isinstance(src, cuda.ndarray):
            cuda.cupy.copyto(dst, src)
        else:
            raise TypeError('cannot copy from non-array object of type {}'
                            .format(type(src)))
    else:
        raise TypeError('cannot copy to non-array object of type {}'.format(
            type(dst)))
