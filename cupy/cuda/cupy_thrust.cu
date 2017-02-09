#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "cupy_common.h"
#include "cupy_thrust.h"

using namespace thrust;


/*
 * sort
 */

template <typename T>
void cupy::thrust::sort(void *start, ssize_t num) {
    device_ptr<T> dp_first = device_pointer_cast((T *)start);
    device_ptr<T> dp_last  = device_pointer_cast((T *)start + num);
    stable_sort< device_ptr<T> >(dp_first, dp_last);
}

template void cupy::thrust::sort<cpy_byte>(void *, ssize_t);
template void cupy::thrust::sort<cpy_ubyte>(void *, ssize_t);
template void cupy::thrust::sort<cpy_short>(void *, ssize_t);
template void cupy::thrust::sort<cpy_ushort>(void *, ssize_t);
template void cupy::thrust::sort<cpy_int>(void *, ssize_t);
template void cupy::thrust::sort<cpy_uint>(void *, ssize_t);
template void cupy::thrust::sort<cpy_long>(void *, ssize_t);
template void cupy::thrust::sort<cpy_ulong>(void *, ssize_t);
template void cupy::thrust::sort<cpy_float>(void *, ssize_t);
template void cupy::thrust::sort<cpy_double>(void *, ssize_t);


/*
 * argsort
 */

template <typename T>
class elem_less {
public:
    elem_less(void *data):_data((const T *)data) {}
    __device__ bool operator()(ssize_t i, ssize_t j) { return _data[i] < _data[j]; }
private:
    const T *_data;
};

template <typename T>
void cupy::thrust::argsort(ssize_t *idx_start, void *data_start, ssize_t num) {
    /* idx_start is the beggining of an output array where the indexes that
       would sort the data will be placed. The original contents of idx_start
       will be destroyed. */
    device_ptr<ssize_t> dp_first = device_pointer_cast(idx_start);
    device_ptr<ssize_t> dp_last  = device_pointer_cast(idx_start + num);
    sequence(dp_first, dp_last);
    stable_sort< device_ptr<ssize_t> >(dp_first, dp_last, elem_less<T>(data_start));
}

template void cupy::thrust::argsort<cpy_byte>(ssize_t *, void *, ssize_t);
template void cupy::thrust::argsort<cpy_ubyte>(ssize_t *, void *, ssize_t);
template void cupy::thrust::argsort<cpy_short>(ssize_t *, void *, ssize_t);
template void cupy::thrust::argsort<cpy_ushort>(ssize_t *, void *, ssize_t);
template void cupy::thrust::argsort<cpy_int>(ssize_t *, void *, ssize_t);
template void cupy::thrust::argsort<cpy_uint>(ssize_t *, void *, ssize_t);
template void cupy::thrust::argsort<cpy_long>(ssize_t *, void *, ssize_t);
template void cupy::thrust::argsort<cpy_ulong>(ssize_t *, void *, ssize_t);
template void cupy::thrust::argsort<cpy_float>(ssize_t *, void *, ssize_t);
template void cupy::thrust::argsort<cpy_double>(ssize_t *, void *, ssize_t);
