#include "xchainer/shape.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "xchainer/axis.h"
#include "xchainer/strides.h"

namespace xchainer {

namespace internal {

bool IsContiguous(const Shape& shape, const Strides& strides, int64_t element_bytes) {
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
        if (*strides_it != element_bytes) {
            return false;
        }
        element_bytes *= *shape_it;
    }
    return true;
}

Shape BroadcastShapes(const Shape& shape0, const Shape& shape1) {
    if (shape0.size() < shape1.size()) {
        return BroadcastShapes(shape1, shape0);
    }
    assert(shape0.size() >= shape1.size());

    std::vector<int64_t> new_dims;
    new_dims.reserve(shape0.size());

    // If shape0 is longer than shape1, they are aligned at the ending position and shape0_mid is aligned to shape1.begin().
    auto shape0_mid = shape0.begin() + (shape0.size() - shape1.size());
    std::copy(shape0.begin(), shape0_mid, std::back_inserter(new_dims));
    std::transform(shape0_mid, shape0.end(), shape1.begin(), std::back_inserter(new_dims), [&shape0, &shape1](int64_t dim0, int64_t dim1) {
        if (dim0 == dim1) {
            return dim0;
        }
        if (dim0 == 1) {
            return dim1;
        }
        if (dim1 == 1) {
            return dim0;
        }
        throw DimensionError("operands could not be broadcast together with shapes " + shape0.ToString() + ' ' + shape1.ToString());
    });

    return Shape{new_dims.begin(), new_dims.end()};
}

bool IsValidReductionShape(const Shape& in_shape, const std::vector<int8_t>& axis, const Shape& out_shape, bool allow_keepdims) {
    return out_shape.ndim() == in_shape.ndim() - static_cast<int64_t>(axis.size()) ||
           (allow_keepdims && out_shape.ndim() == in_shape.ndim());
}

Shape TransposeShape(const Shape& shape, const std::vector<int8_t>& axes) {
    assert(IsAxesPermutation(axes, shape.ndim()));
    std::vector<int64_t> new_shape;
    new_shape.reserve(shape.ndim());
    for (int8_t axis : axes) {
        new_shape.push_back(shape[axis]);
    }
    return Shape{new_shape.begin(), new_shape.end()};
}

}  // namespace internal

int64_t Shape::GetTotalSize() const {
    const auto first = dims_.begin();
    const auto last = first + ndim_;
    auto total_size = std::accumulate(first, last, int64_t{1}, std::multiplies<>());
    return total_size;
}

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
        throw DimensionError("Shapes do not match: " + lhs.ToString() + ", " + rhs.ToString() + ".");
    }
}

}  // namespace xchainer
