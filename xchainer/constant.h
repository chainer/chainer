#pragma once

namespace xchainer {

// Maximum number of dimensions (axes) of each array.
constexpr int8_t kMaxNdim = 10;

// Reserved dimension for dynamic-length arrays.
constexpr int8_t kDynamicNdim = -1;

// Default backprop ID to be used if not specified
constexpr const char* kDefaultGraphName = "default";

}  // namespace xchainer
