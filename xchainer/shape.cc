#include "xchainer/shape.h"

#include <functional>
#include <numeric>

namespace xchainer {

int64_t Shape::TotalSize() const {
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
        throw DimensionError("shape mismatched");
    }
}

}  // namespace xchainer
