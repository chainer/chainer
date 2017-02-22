#ifndef INCLUDE_GUARD_CUPY_CUDA_THRUST_H
#define INCLUDE_GUARD_CUPY_CUDA_THRUST_H

#ifndef CUPY_NO_CUDA

namespace cupy {

namespace thrust {

template <typename T> void sort(void *, ssize_t);

template <typename T> void lexsort(ssize_t *, void *, ssize_t, ssize_t);

template <typename T> void argsort(ssize_t *, void *, ssize_t);

} // namespace thrust

} // namespace cupy

#else // CUPY_NO_CUDA

#include "cupy_common.h"

namespace cupy {

namespace thrust {

template <typename T> void sort(void *, ssize_t) { return; }

template <typename T> void lexsort(ssize_t *, void *, ssize_t, ssize_t) { return; }

template <typename T> void argsort(ssize_t *, void *, ssize_t) { return; }

} // namespace thrust

} // namespace cupy

#endif // #ifndef CUPY_NO_CUDA

#endif // INCLUDE_GUARD_CUPY_CUDA_THRUST_H
