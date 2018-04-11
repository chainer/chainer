#pragma once

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

namespace xchainer {
namespace internal {

// Normalizes possibly-negative axis to non-negative axis in [0, ndim).
// If `axis` does not fit in [-ndim, ndim), DimensionError is thrown.
int8_t NormalizeAxis(int8_t axis, int8_t ndim);

// Resolves the axis argument of many operations.
// Negative axis value is first converted to non-negative one (by wrapping at ndim), and then the axis is sorted.
// In GetSortedAxesOrAll, nullopt is converted to a vector of all axes.
std::vector<int8_t> GetSortedAxes(const std::vector<int8_t>& axis, int8_t ndim);
std::vector<int8_t> GetSortedAxesOrAll(const nonstd::optional<std::vector<int8_t>>& axis, int8_t ndim);

}  // namespace internal
}  // namespace xchainer
