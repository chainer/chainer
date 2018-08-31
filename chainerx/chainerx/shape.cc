#include "chainerx/shape.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>

#include "chainerx/axes.h"
#include "chainerx/macro.h"
#include "chainerx/strides.h"

namespace chainerx {
namespace internal {

bool IsContiguous(const Shape& shape, const Strides& strides, int64_t item_size) {
    CHAINERX_ASSERT(shape.size() == strides.size());
    int64_t total_size = shape.GetTotalSize();
    if (total_size == 0 || total_size == 1) {
        return true;
    }
    auto shape_it = shape.rbegin();
    for (auto strides_it = strides.rbegin(); strides_it != strides.rend(); ++shape_it, ++strides_it) {
        if (*shape_it == 1) {
            continue;
        }
        if (*strides_it != item_size) {
            return false;
        }
        item_size *= *shape_it;
    }
    return true;
}

bool IsValidReductionShape(const Shape& in_shape, const Axes& axes, const Shape& out_shape, bool allow_keepdims) {
    return out_shape.ndim() == in_shape.ndim() - static_cast<int64_t>(axes.size()) ||
           (allow_keepdims && out_shape.ndim() == in_shape.ndim());
}

int64_t CountItemsAlongAxes(const Shape& shape, const Axes& axes) {
    return std::accumulate(axes.begin(), axes.end(), int64_t{1}, [&shape](int64_t count, int8_t i) { return count * shape[i]; });
}

Shape BroadcastShapes(const Shape& shape0, const Shape& shape1) {
    if (shape0.size() < shape1.size()) {
        return BroadcastShapes(shape1, shape0);
    }
    CHAINERX_ASSERT(shape0.size() >= shape1.size());

    Shape new_shape;

    // If shape0 is longer than shape1, they are aligned at the ending position and shape0_mid is aligned to shape1.begin().
    auto shape0_mid = shape0.begin() + (shape0.size() - shape1.size());
    std::copy(shape0.begin(), shape0_mid, std::back_inserter(new_shape));
    std::transform(shape0_mid, shape0.end(), shape1.begin(), std::back_inserter(new_shape), [&shape0, &shape1](int64_t dim0, int64_t dim1) {
        if (dim0 == dim1) {
            return dim0;
        }
        if (dim0 == 1) {
            return dim1;
        }
        if (dim1 == 1) {
            return dim0;
        }
        throw DimensionError{"operands could not be broadcast together with shapes ", shape0, " ", shape1};
    });

    return new_shape;
}

Shape ReduceShape(const Shape& shape, const Axes& axes, bool keepdims) {
    CHAINERX_ASSERT(shape.ndim() >= axes.ndim());
    Shape reduced;
    int8_t i_axis = 0;
    for (int8_t i = 0; i < shape.ndim(); ++i) {
        if (i_axis < axes.ndim() && i == axes[i_axis]) {
            ++i_axis;
            if (keepdims) {
                reduced.emplace_back(int64_t{1});
            }
        } else {
            reduced.emplace_back(shape[i]);
        }
    }
    CHAINERX_ASSERT(i_axis == axes.ndim());
    CHAINERX_ASSERT(reduced.ndim() == shape.ndim() - static_cast<int8_t>(!keepdims) * axes.ndim());
    return reduced;
}

Shape ExpandShape(const Shape& shape, const Axes& axes) {
    Shape expanded;
    int8_t i_axis = 0;
    int8_t i_shape = 0;
    int8_t reduced_ndim = shape.ndim() + axes.ndim();
    for (int8_t i = 0; i < reduced_ndim; ++i) {
        if (i_axis < axes.ndim() && i == axes[i_axis]) {
            expanded.emplace_back(int64_t{1});
            ++i_axis;
        } else {
            expanded.emplace_back(shape[i_shape]);
            ++i_shape;
        }
    }
    CHAINERX_ASSERT(i_axis == axes.ndim());
    CHAINERX_ASSERT(i_shape == shape.ndim());
    CHAINERX_ASSERT(expanded.ndim() == shape.ndim() + axes.ndim());
    return expanded;
}

Shape TransposeShape(const Shape& shape, const Axes& axes) {
    CHAINERX_ASSERT(IsAxesPermutation(axes, shape.ndim()));
    Shape new_shape;
    for (int8_t axis : axes) {
        new_shape.emplace_back(shape[axis]);
    }
    return new_shape;
}

}  // namespace internal

int64_t Shape::GetTotalSize() const { return std::accumulate(begin(), end(), int64_t{1}, std::multiplies<>()); }

std::string Shape::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "(";
    for (auto iter = shape.begin(); iter != shape.end(); ++iter) {
        if (iter != shape.begin()) {
            os << ", ";
        }
        os << *iter;
    }
    // same as Python tuples with trailing comma in case of length 1
    return os << (shape.ndim() == 1 ? ",)" : ")");
}

void CheckEqual(const Shape& lhs, const Shape& rhs) {
    if (lhs != rhs) {
        throw DimensionError{"Shapes do not match: ", lhs, ", ", rhs, "."};
    }
}

}  // namespace chainerx
