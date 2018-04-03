#include "xchainer/routines/sorting.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/util.h"

namespace xchainer {

Array ArgMax(const Array& a, const nonstd::optional<int8_t>& axis) {
    std::vector<int8_t> sorted_axis;
    std::vector<int64_t> out_shape_vec;
    if (axis.has_value()) {
        sorted_axis = internal::GetSortedAxes({*axis}, a.ndim());
        out_shape_vec.reserve(a.ndim() - 1);
        for (int8_t i = 0; i < a.ndim(); ++i) {
            if (i != *axis) {
                out_shape_vec.push_back(a.shape()[i]);
            }
        }
    } else {
        // Fill with all axes
        sorted_axis.resize(a.ndim());
        std::iota(sorted_axis.begin(), sorted_axis.end(), int8_t{0});
        out_shape_vec.push_back(1);
    }

    Array out = Empty({out_shape_vec.begin(), out_shape_vec.end()}, Dtype::kInt64, a.device());
    a.device().ArgMax(a, sorted_axis, out);
    return out;
}

}  // namespace xchainer
