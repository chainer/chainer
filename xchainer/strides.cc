#include "xchainer/strides.h"

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/axes.h"
#include "xchainer/error.h"
#include "xchainer/shape.h"

namespace xchainer {

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

namespace internal {

Strides ExpandStrides(const Strides& strides, const Axes& axes) {
    assert(strides.ndim() >= axes.ndim());
    Strides expanded;
    int8_t out_ndim = strides.ndim() + axes.ndim();
    int8_t i_axis = 0;
    int8_t i_stride = 0;
    for (int8_t i = 0; i < out_ndim; ++i) {
        if (i_axis < axes.ndim() && i == axes[i_axis]) {
            expanded.emplace_back(int64_t{1});
            ++i_axis;
        } else {
            expanded.emplace_back(strides[i_stride]);
            ++i_stride;
        }
    }
    assert(i_axis == axes.ndim());
    assert(i_stride == strides.ndim());
    assert(expanded.ndim() == strides.ndim() + axes.ndim());
    return expanded;
}

Strides BroadcastStrides(const Strides& strides, const Shape& in_shape, const Shape& out_shape) {
    assert(strides.ndim() == in_shape.ndim());
    assert(strides.ndim() <= out_shape.ndim());
    Strides broadcasted;
    broadcasted.resize(out_shape.ndim());
    int8_t i_in = in_shape.ndim() - 1;
    for (int8_t i_out = out_shape.ndim() - 1; i_out >= 0; --i_out) {
        int64_t out_dim = out_shape[i_out];

        // If this dimension is to be broadcasted, nonbroadcast_stride is unset.
        // Otherwise, it holds the new stride.
        nonstd::optional<int64_t> nonbroadcast_stride{};

        if (i_in >= 0) {
            int64_t in_dim = in_shape[i_in];
            if (in_dim == 1) {
                // do nothing; broadcast
            } else if (in_dim == out_dim) {
                nonbroadcast_stride = strides[i_in];
            } else {
                throw DimensionError{"Invalid broadcast from ", in_shape, " to ", out_shape};
            }
            --i_in;
        } else {
            // do nothing; broadcast
        }

        if (nonbroadcast_stride.has_value()) {
            // non-broadcast dimension
            broadcasted[i_out] = nonbroadcast_stride.value();
        } else {
            // broadcast dimension
            broadcasted[i_out] = int64_t{0};
        }
    }
    assert(i_in == -1);
    assert(broadcasted.ndim() == out_shape.ndim());
    return broadcasted;
}

}  // namespace internal

void CheckEqual(const Strides& lhs, const Strides& rhs) {
    if (lhs != rhs) {
        throw DimensionError{"strides mismatched"};
    }
}

std::tuple<int64_t, int64_t> GetDataRange(const Shape& shape, const Strides& strides, size_t item_size) {
    assert(shape.ndim() == strides.ndim());
    int64_t first = 0;
    int64_t last = 0;

    for (int8_t i = 0; i < shape.ndim(); ++i) {
        auto& first_or_last = strides[i] < 0 ? first : last;
        first_or_last += shape[i] * strides[i];
    }
    assert(first <= 0);
    assert(0 <= last);
    return std::tuple<int64_t, int64_t>{first, last + item_size};
}

}  // namespace xchainer
