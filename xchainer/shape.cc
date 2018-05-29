#include "xchainer/shape.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <string>

#include "xchainer/axes.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace internal {

bool IsContiguous(const Shape& shape, const Strides& strides, int64_t item_size) {
    assert(shape.size() == strides.size());
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

bool IsValidReductionShape(const Shape& in_shape, const Axes& axis, const Shape& out_shape, bool allow_keepdims) {
    return out_shape.ndim() == in_shape.ndim() - static_cast<int64_t>(axis.size()) ||
           (allow_keepdims && out_shape.ndim() == in_shape.ndim());
}

Shape BroadcastShapes(const Shape& shape0, const Shape& shape1) {
    if (shape0.size() < shape1.size()) {
        return BroadcastShapes(shape1, shape0);
    }
    assert(shape0.size() >= shape1.size());

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

Shape ReduceShape(const Shape& in_shape, const Axes& axes, bool keepdims) {
    assert(in_shape.ndim() >= axes.ndim());
    Shape out_shape;
    auto axes_size = static_cast<int8_t>(axes.size());
    int8_t i_axis = 0;
    for (int8_t i = 0; i < in_shape.ndim(); ++i) {
        if (i_axis < axes_size && i == axes[i_axis]) {
            ++i_axis;
            if (keepdims) {
                out_shape.emplace_back(int64_t{1});
            }
        } else {
            out_shape.emplace_back(in_shape[i]);
        }
    }
    return out_shape;
}

Shape TransposeShape(const Shape& shape, const Axes& axes) {
    assert(IsAxesPermutation(axes, shape.ndim()));
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

}  // namespace xchainer
