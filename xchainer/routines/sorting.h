#pragma once

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"

namespace xchainer {

Array ArgMax(const Array& a, const nonstd::optional<std::vector<int8_t>>& axis = nonstd::nullopt);

}  // namespace xchainer
