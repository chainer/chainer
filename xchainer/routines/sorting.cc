#include "xchainer/routines/sorting.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/util.h"

namespace xchainer {

Array ArgMax(const Array& a, const nonstd::optional<int8_t>& axis) {
    if (a.GetTotalSize() == 0) {
        throw DimensionError("Cannot compute ArgMax for an empty array.");
    }

    std::vector<int8_t> sorted_axis;
    std::vector<int64_t> out_shape_vec;
    if (axis.has_value()) {
        sorted_axis = internal::GetSortedAxes({*axis}, a.ndim());
        out_shape_vec.reserve(a.ndim() - 1);
        int8_t i_axis = 0;
        for (int8_t i = 0; i < a.ndim(); ++i) {
            if (i_axis < static_cast<int8_t>(sorted_axis.size()) && i == sorted_axis[i_axis]) {
                ++i_axis;
            } else {
                out_shape_vec.push_back(a.shape()[i]);
            }
        }
    } else {
        // Fill with all axes
        sorted_axis.resize(a.ndim());
        std::iota(sorted_axis.begin(), sorted_axis.end(), int8_t{0});
    }

    Array out = Empty({out_shape_vec.begin(), out_shape_vec.end()}, Dtype::kInt64, a.device());
    a.device().ArgMax(a, sorted_axis, out);
    return out;
}

}  // namespace xchainer
