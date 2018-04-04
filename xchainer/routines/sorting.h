#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"

namespace xchainer {

Array ArgMax(const Array& a, const nonstd::optional<int8_t>& axis = nonstd::nullopt);

}  // namespace xchainer
