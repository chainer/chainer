#include "xchainer/routines/util.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "xchainer/error.h"

namespace xchainer {
namespace internal {

int8_t NormalizeAxis(int8_t axis, int8_t ndim) {
    if (axis < -ndim || ndim <= axis) {
        throw DimensionError("Axis " + std::to_string(axis) + " is out of bounds for array of dimension " + std::to_string(ndim));
    }
    if (axis < 0) {
        return axis + ndim;
    }
    return axis;
}

std::vector<int8_t> GetSortedAxes(const std::vector<int8_t>& axis, int8_t ndim) {
    std::vector<int8_t> sorted_axis = axis;

    for (auto& a : sorted_axis) {
        a = NormalizeAxis(a, ndim);
    }
    std::sort(sorted_axis.begin(), sorted_axis.end());
    if (std::unique(sorted_axis.begin(), sorted_axis.end()) != sorted_axis.end()) {
        throw DimensionError("Duplicate axis values.");
    }

    // sorted_axis is sorted, unique, and within bounds [0, ndim).
    assert(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));
    assert(std::set<int8_t>(sorted_axis.begin(), sorted_axis.end()).size() == sorted_axis.size());
    assert(std::all_of(sorted_axis.begin(), sorted_axis.end(), [ndim](int8_t x) -> bool { return 0 <= x && x < ndim; }));
    return sorted_axis;
}

std::vector<int8_t> GetSortedAxesOrAll(const nonstd::optional<std::vector<int8_t>>& axis, int8_t ndim) {
    if (axis.has_value()) {
        return GetSortedAxes(*axis, ndim);
    }
    // Fill with all axes
    std::vector<int8_t> sorted_axis;
    sorted_axis.resize(ndim);
    std::iota(sorted_axis.begin(), sorted_axis.end(), int8_t{0});
    return sorted_axis;
}

}  // namespace internal
}  // namespace xchainer
