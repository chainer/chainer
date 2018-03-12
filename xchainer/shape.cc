#include "xchainer/shape.h"

#include <functional>
#include <numeric>

#include "xchainer/strides.h"

namespace xchainer {

namespace internal {

bool IsContiguous(const Shape& shape, const Strides& strides, int64_t element_bytes) {
    Expects(shape.size() == strides.size());
    auto shape_it = shape.rbegin();
    for (auto strides_it = strides.rbegin(); strides_it != strides.rend(); ++shape_it, ++strides_it) {
        if (*strides_it != element_bytes) {
            return false;
        }
        element_bytes *= *shape_it;
    }
    return true;
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
        throw DimensionError("Shapes does not match: " + lhs.ToString() + ", " + rhs.ToString() + ".");
    }
}

}  // namespace xchainer
