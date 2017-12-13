#include "xchainer/shape.h"

#include <functional>
#include <numeric>

namespace xchainer {

int64_t Shape::total_size() const {
    const auto first = dims_.begin();
    const auto last = first + ndim_;
    auto total_size = std::accumulate(first, last, static_cast<int64_t>(1), std::multiplies<>());
    return total_size;
}

void CheckEqual(const Shape& lhs, const Shape& rhs) {
    if (lhs != rhs) {
        throw DimensionError("shape mismatched");
    }
}

}  // namespace xchainer
