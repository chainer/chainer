#pragma once

#include "chainerx/cuda/float16.cuh"
#include "chainerx/dtype.h"
#include "chainerx/float16.h"

namespace chainerx {
namespace cuda {
namespace data_type_detail {

template <typename T>
struct DataTypeSpec {
    using DataType = T;
};

template <>
struct DataTypeSpec<chainerx::Float16> {
    using DataType = cuda::Float16;
};

template <>
struct DataTypeSpec<const chainerx::Float16> {
    using DataType = const cuda::Float16;
};

}  // namespace data_type_detail

namespace cuda_internal {

template <typename T>
using DataType = typename data_type_detail::DataTypeSpec<T>::DataType;

template <typename T>
using StorageType = TypeToDeviceStorageType<T>;

template <typename T>
__host__ __device__ DataType<T>& StorageToDataType(StorageType<T>& x) {
    return *reinterpret_cast<DataType<T>*>(&x);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}
template <typename T>
__host__ __device__ DataType<T> StorageToDataType(StorageType<T>&& x) {
    return *reinterpret_cast<DataType<T>*>(&x);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}
template <typename T>
__host__ __device__ StorageType<T>& DataToStorageType(DataType<T>& x) {
    return *reinterpret_cast<StorageType<T>*>(&x);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}
template <typename T>
__host__ __device__ StorageType<T> DataToStorageType(DataType<T>&& x) {
    return *reinterpret_cast<StorageType<T>*>(&x);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
