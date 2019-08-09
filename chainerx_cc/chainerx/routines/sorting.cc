#include "chainerx/routines/sorting.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/kernels/sorting.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/reduction.h"
#include "chainerx/shape.h"

namespace chainerx {

Array ArgMax(const Array& a, const OptionalAxes& axis) {
    Axes sorted_axis{};
    Shape out_shape{};
    if (axis.has_value()) {
        sorted_axis = internal::GetSortedAxes(*axis, a.ndim());
        int8_t i_axis = 0;
        for (int8_t i = 0; i < a.ndim(); ++i) {
            if (i_axis < static_cast<int8_t>(sorted_axis.size()) && i == sorted_axis[i_axis]) {
                ++i_axis;
            } else {
                out_shape.emplace_back(a.shape()[i]);
            }
        }
    } else {
        // Fill with all axes
        sorted_axis.resize(a.ndim());
        std::iota(sorted_axis.begin(), sorted_axis.end(), int8_t{0});
    }

    for (int8_t i : sorted_axis) {
        if (a.shape()[i] == 0) {
            throw DimensionError{"Cannot compute ArgMax for an empty array."};
        }
    }

    Array out = Empty(out_shape, Dtype::kInt64, a.device());
    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<ArgMaxKernel>(a, sorted_axis, out);
    }
    return out;
}

Array ArgMin(const Array& a, const OptionalAxes& axis) {
    Axes sorted_axis{};
    Shape out_shape{};
    if (axis.has_value()) {
        sorted_axis = internal::GetSortedAxes(*axis, a.ndim());
        int8_t i_axis = 0;
        for (int8_t i = 0; i < a.ndim(); ++i) {
            if (i_axis < static_cast<int8_t>(sorted_axis.size()) && i == sorted_axis[i_axis]) {
                ++i_axis;
            } else {
                out_shape.emplace_back(a.shape()[i]);
            }
        }
    } else {
        // Fill with all axes
        sorted_axis.resize(a.ndim());
        std::iota(sorted_axis.begin(), sorted_axis.end(), int8_t{0});
    }

    for (int8_t i : sorted_axis) {
        if (a.shape()[i] == 0) {
            throw DimensionError{"Cannot compute ArgMin for an empty array."};
        }
    }

    Array out = Empty(out_shape, Dtype::kInt64, a.device());
    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<ArgMinKernel>(a, sorted_axis, out);
    }
    return out;
}

Array CountNonzero(const Array& a, const OptionalAxes& axis) {
    // TODO(aksub99): Fix after NotEqual(Array, Scalar) is supported.
    Array out = (a != ZerosLike(a)).Sum(axis);
    return out;
}

Array NanArgMax(const Array& a, const OptionalAxes& axis) {
    Axes sorted_axis{};
    Shape out_shape{};
    Array a_replaced = Where(IsNan(a), -INFINITY, a);
    if (axis.has_value()) {
        sorted_axis = internal::GetSortedAxes(*axis, a_replaced.ndim());
        int8_t i_axis = 0;
        for (int8_t i = 0; i < a.ndim(); ++i) {
            if (i_axis < static_cast<int8_t>(sorted_axis.size()) && i == sorted_axis[i_axis]) {
                ++i_axis;
            } else {
                out_shape.emplace_back(a_replaced.shape()[i]);
            }
        }
    } else {
        // Fill with all axes
        sorted_axis.resize(a_replaced.ndim());
        std::iota(sorted_axis.begin(), sorted_axis.end(), int8_t{0});
    }

    for (int8_t i : sorted_axis) {
        if (a_replaced.shape()[i] == 0) {
            throw DimensionError{"Cannot compute NanArgMax for an empty array."};
        }
    }

    Array out = Empty(out_shape, Dtype::kInt64, a_replaced.device());
    {
        NoBackpropModeScope scope{};
        a_replaced.device().backend().CallKernel<NanArgMaxKernel>(a_replaced, sorted_axis, out);
    }
    return out;
}

Array NanArgMin(const Array& a, const OptionalAxes& axis) {
    Axes sorted_axis{};
    Shape out_shape{};
    Array a_replaced = Where(IsNan(a), INFINITY, a);

    if (axis.has_value()) {
        sorted_axis = internal::GetSortedAxes(*axis, a_replaced.ndim());
        int8_t i_axis = 0;
        for (int8_t i = 0; i < a_replaced.ndim(); ++i) {
            if (i_axis < static_cast<int8_t>(sorted_axis.size()) && i == sorted_axis[i_axis]) {
                ++i_axis;
            } else {
                out_shape.emplace_back(a_replaced.shape()[i]);
            }
        }
    } else {
        // Fill with all axes
        sorted_axis.resize(a_replaced.ndim());
        std::iota(sorted_axis.begin(), sorted_axis.end(), int8_t{0});
    }

    for (int8_t i : sorted_axis) {
        if (a_replaced.shape()[i] == 0) {
            throw DimensionError{"Cannot compute NanArgMin for an empty array."};
        }
    }

    Array out = Empty(out_shape, Dtype::kInt64, a_replaced.device());
    {
        NoBackpropModeScope scope{};
        a_replaced.device().backend().CallKernel<NanArgMinKernel>(a_replaced, sorted_axis, out);
    }
    return out;
}

}  // namespace chainerx
