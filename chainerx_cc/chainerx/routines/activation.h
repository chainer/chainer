#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array Sigmoid(const Array& x);

Array Relu(const Array& x);

}  // namespace chainerx
