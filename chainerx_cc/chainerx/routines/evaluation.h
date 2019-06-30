#pragma once

#include <cstdint>
#include <string>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"

namespace chainerx {

Array Accuracy(const Array& x1, const Array& x2, const nonstd::optional<Array>& ignore_label);

}  // namespace chainerx
