#pragma once

#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_index.h"

namespace xchainer {
namespace routines {
namespace internal {

// Returns a view selected with the indices.
Array At(const Array& a, const std::vector<ArrayIndex>& indices);

}  // namespace internal
}  // namespace routines
}  // namespace xchainer
