#include "chainerx/strides.h"

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>

#include <gsl/gsl>

#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"

namespace chainerx {

Strides::Strides(const Shape& shape, int64_t item_size) {
    int64_t stride = item_size;
    int8_t ndim = shape.ndim();
    resize(ndim);
    auto it = rbegin();
    for (int8_t i = ndim - 1; i >= 0; --i, ++it) {
        *it = stride;
        stride *= std::max(int64_t{1}, shape[i]);
    }
}

std::string Strides::ToString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const Strides& strides) {
    os << "(";
    for (auto iter = strides.begin(); iter != strides.end(); ++iter) {
        if (iter != strides.begin()) {
            os << ", ";
        }
        os << *iter;
    }
    // same as Python tuples with trailing comma in case of length 1
    return os << (strides.ndim() == 1 ? ",)" : ")");
}

void CheckEqual(const Strides& lhs, const Strides& rhs) {
    if (lhs != rhs) {
        throw DimensionError{"strides mismatched"};
    }
}

std::tuple<int64_t, int64_t> GetDataRange(const Shape& shape, const Strides& strides, size_t item_size) {
    CHAINERX_ASSERT(shape.ndim() == strides.ndim());
    int64_t first = 0;
    int64_t last = item_size;

    for (int8_t i = 0; i < shape.ndim(); ++i) {
        auto& first_or_last = strides[i] < 0 ? first : last;
        if (shape[i] == 0) {
            return std::tuple<int64_t, int64_t>{0, 0};
        }
        first_or_last += (shape[i] - 1) * strides[i];
    }
    CHAINERX_ASSERT(first <= 0);
    CHAINERX_ASSERT(static_cast<int64_t>(item_size) <= last);
    return std::tuple<int64_t, int64_t>{first, last};
}

}  // namespace chainerx
