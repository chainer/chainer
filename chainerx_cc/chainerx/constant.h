#pragma once

namespace chainerx {

// Maximum number of dimensions (axes) of each array.
constexpr int8_t kMaxNdim = 10;

// Reserved dimension for dynamic-length arrays.
constexpr int8_t kDynamicNdim = -1;

// Default backprop ID to be used if not specified
constexpr const char* kDefaultBackpropName = "default";

// Value of constant PI
constexpr long double kPi = 3.141592653589793238462643383279502884L;

}  // namespace chainerx
