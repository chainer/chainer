#pragma once

#include <cstdint>

#include "xchainer/array.h"

namespace xchainer {
namespace internal {

template <typename T>
T* GetRawOffsetData(const Array& a) {
    uint8_t* offset_ptr = static_cast<uint8_t*>(a.raw_data()) + a.offset();
    return reinterpret_cast<T*>(offset_ptr);  // NOLINT(readability/nolint), NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

}  // namespace internal
}  // namespace xchainer
