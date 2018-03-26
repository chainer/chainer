#pragma once

#include <vector>

#include "xchainer/array_index.h"

namespace xchainer {

class Array;

namespace routines {

// Returns a view selected with the indices.
Array At(const Array& array, const std::vector<ArrayIndex>& indices);

}  // namespace routines
}  // namespace xchaienr


