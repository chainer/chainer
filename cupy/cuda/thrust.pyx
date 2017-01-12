# distutils: language = c++

"""Thin wrapper of Thrust implementations for CuPy API."""

###############################################################################
# Extern
###############################################################################

cdef extern from "../cuda/cupy_thrust.h" namespace "cupy::thrust":
    void sort[T](void *start, ssize_t num)
    void argsort[T](ssize_t *idx_start, void *data_start, ssize_t num)
