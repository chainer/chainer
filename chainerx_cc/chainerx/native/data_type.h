#pragma once

#include <cstdint>
#include <type_traits>

#include "chainerx/dtype.h"

namespace chainerx {
namespace native {
namespace native_internal {

template <typename T>
using StorageType = TypeToDeviceStorageType<T>;

template <typename T>
T& StorageToDataType(StorageType<T>& x) {
    return *reinterpret_cast<T*>(&x);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}
template <typename T>
StorageType<T>& DataToStorageType(T& x) {
    return *reinterpret_cast<StorageType<T>*>(&x);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}
template <typename T>
T StorageToDataType(StorageType<T>&& x) {
    return *reinterpret_cast<T*>(&x);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}
template <typename T>
StorageType<T> DataToStorageType(T&& x) {
    return *reinterpret_cast<StorageType<T>*>(&x);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

}  // namespace native_internal

// This function is used from outside of native namespace.
template <typename T>
T& StorageToDataType(native_internal::StorageType<T>& x) {
    return native_internal::StorageToDataType<T>(x);
}

// This function is used from outside of native namespace.
template <typename T>
T StorageToDataType(native_internal::StorageType<T>&& x) {
    return native_internal::StorageToDataType<T>(x);
}

}  // namespace native
}  // namespace chainerx
