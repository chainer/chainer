#pragma once

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/ndim_vector.h"

namespace xchainer {
namespace internal {

// Normalizes possibly-negative axis to non-negative axis in [0, ndim).
// If `axis` does not fit in [-ndim, ndim), DimensionError is thrown.
int8_t NormalizeAxis(int8_t axis, int8_t ndim);

// Resolves the axis argument of many operations.
// Negative axis value is first converted to non-negative one (by wrapping at ndim), and then the axis is sorted.
// In GetSortedAxesOrAll, nullopt is converted to a vector of all axes.
NdimVector<int8_t> GetSortedAxes(const NdimVector<int8_t>& axis, int8_t ndim);
NdimVector<int8_t> GetSortedAxesOrAll(const nonstd::optional<NdimVector<int8_t>>& axis, int8_t ndim);

}  // namespace internal
}  // namespace xchainer
