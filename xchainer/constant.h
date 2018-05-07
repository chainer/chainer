#pragma once

namespace xchainer {

// Maximum number of dimensions (axes) of each array.
constexpr int8_t kMaxNdim = 8;

// Reserved dimension for dynamic-length arrays.
constexpr int8_t kDynamicNdim = -1;

// Default graph ID to be used if not specified
constexpr const char* kDefaultGraphId = "default";

}  // namespace xchainer
