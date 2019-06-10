#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array Sigmoid(const Array& x);

Array Relu(const Array& x);

Array IsNan(const Array& x);

Array IsInf(const Array& x);

Array IsFinite(const Array& x);

Array Ceil(const Array& x);

Array Floor(const Array& x);

}  // namespace chainerx
