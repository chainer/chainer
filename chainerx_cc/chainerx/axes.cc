#include "chainerx/axes.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "chainerx/macro.h"

namespace chainerx {

std::string Axes::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const Axes& axes) {
    os << "(";
    for (auto iter = axes.begin(); iter != axes.end(); ++iter) {
        if (iter != axes.begin()) {
            os << ", ";
        }
        os << static_cast<int>(*iter);
    }
    // same as Python tuples with trailing comma in case of length 1
    return os << (axes.ndim() == 1 ? ",)" : ")");
}

namespace internal {

bool IsAxesPermutation(const Axes& axes, int8_t ndim) {
    CHAINERX_ASSERT(ndim >= 0);
    if (axes.size() != static_cast<size_t>(ndim)) {
        return false;
    }

    Axes sorted_axes = axes;
    std::sort(sorted_axes.begin(), sorted_axes.end());
    for (int8_t i = 0; i < ndim; ++i) {
        if (sorted_axes[i] != i) {
            return false;
        }
    }
    return true;
}

int8_t NormalizeAxis(int8_t axis, int8_t ndim) {
    if (axis < -ndim || ndim <= axis) {
        throw DimensionError{"Axis ", axis, " is out of bounds for array of dimension ", ndim};
    }
    if (axis < 0) {
        return axis + ndim;
    }
    return axis;
}

Axes GetNormalizedAxes(const Axes& axis, int8_t ndim) {
    Axes normalized_axis = axis;

    for (auto& a : normalized_axis) {
        a = NormalizeAxis(a, ndim);
    }
    if (std::unique(normalized_axis.begin(), normalized_axis.end()) != normalized_axis.end()) {
        throw DimensionError{"Duplicate axis values."};
    }

    // normalized_axis is unique, and within bounds [0, ndim).
    CHAINERX_ASSERT(std::set<int8_t>(normalized_axis.begin(), normalized_axis.end()).size() == normalized_axis.size());
    CHAINERX_ASSERT(std::all_of(normalized_axis.begin(), normalized_axis.end(), [ndim](int8_t x) -> bool { return 0 <= x && x < ndim; }));
    return normalized_axis;
}

Axes GetSortedAxes(const Axes& axis, int8_t ndim) {
    Axes sorted_axis = GetNormalizedAxes(axis, ndim);
    std::sort(sorted_axis.begin(), sorted_axis.end());

    // sorted_axis is sorted, unique, and within bounds [0, ndim).
    CHAINERX_ASSERT(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));
    CHAINERX_ASSERT(std::set<int8_t>(sorted_axis.begin(), sorted_axis.end()).size() == sorted_axis.size());
    CHAINERX_ASSERT(std::all_of(sorted_axis.begin(), sorted_axis.end(), [ndim](int8_t x) -> bool { return 0 <= x && x < ndim; }));
    return sorted_axis;
}

Axes GetSortedAxesOrAll(const OptionalAxes& axis, int8_t ndim) {
    if (axis.has_value()) {
        return GetSortedAxes(*axis, ndim);
    }
    // Fill with all axes
    Axes sorted_axis{};
    sorted_axis.resize(ndim);
    std::iota(sorted_axis.begin(), sorted_axis.end(), int8_t{0});
    return sorted_axis;
}

}  // namespace internal
}  // namespace chainerx
