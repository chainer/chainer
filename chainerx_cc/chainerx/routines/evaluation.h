#pragma once

#include <cstdint>

#include <absl/types/optional.h>

#include "chainerx/array.h"

namespace chainerx {

Array Accuracy(const Array& y, const Array& t, absl::optional<int64_t> ignore_label);

}  // namespace chainerx
