#pragma once

#include <vector>

#include "xchainer/array_index.h"

namespace xchainer {

class Array;

namespace routines {

// Returns an array where elements at indices are added by the addends.
//
// The original values of this array are not altered.
Array AddAt(const Array& array, const std::vector<ArrayIndex>& indices, const Array& addend);

}  // namespace routines
}  // namespace xchaienr
