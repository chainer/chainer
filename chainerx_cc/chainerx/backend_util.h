#pragma once

#include <cstdint>

#include "chainerx/array.h"

namespace chainerx {
namespace internal {

inline void* GetRawOffsetData(const Array& a) {
    uint8_t* offset_ptr = static_cast<uint8_t*>(a.raw_data()) + a.offset();
    return offset_ptr;
}

}  // namespace internal
}  // namespace chainerx
