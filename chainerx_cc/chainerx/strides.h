#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <string>
#include <tuple>

#include <absl/types/span.h>
#include <gsl/gsl>

#include "chainerx/axes.h"
#include "chainerx/constant.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"

namespace chainerx {

class Strides : public Dims {
    using BaseVector = Dims;

public:
    using const_iterator = BaseVector::const_iterator;
    using const_reverse_iterator = BaseVector::const_reverse_iterator;

    Strides() = default;

    ~Strides() = default;

    // Creates strides for contiguous array.
    Strides(const Shape& shape, Dtype dtype) : Strides{shape, GetItemSize(dtype)} {}
    Strides(const Shape& shape, int64_t item_size);

    // by iterators
    template <typename InputIt>
    Strides(InputIt first, InputIt last) {
        if (std::distance(first, last) > kMaxNdim) {
            throw DimensionError{"too many dimensions: ", std::distance(first, last)};
        }
        insert(begin(), first, last);
    }

    // by span
    explicit Strides(absl::Span<const int64_t> dims) : Strides{dims.begin(), dims.end()} {}

    // by initializer list
    Strides(std::initializer_list<int64_t> dims) : Strides{dims.begin(), dims.end()} {}

    // copy
    Strides(const Strides&) = default;
    Strides& operator=(const Strides&) = default;

    // move
    Strides(Strides&&) = default;
    Strides& operator=(Strides&&) = default;

    std::string ToString() const;

    int8_t ndim() const noexcept { return gsl::narrow_cast<int8_t>(size()); }

    const int64_t& operator[](int8_t index) const {
        if (!(0 <= index && static_cast<size_t>(index) < size())) {
            throw DimensionError{"Stride index ", index, " out of bounds for strides with ", size(), " size."};
        }
        return this->StackVector::operator[](index);
    }

    int64_t& operator[](int8_t index) {
        if (!(0 <= index && static_cast<size_t>(index) < size())) {
            throw DimensionError{"Stride index ", index, " out of bounds for strides with ", size(), " size."};
        }
        return this->StackVector::operator[](index);
    }

    // span
    absl::Span<const int64_t> span() const { return {*this}; }

    // Rearranges strides in the order specified by the axes.
    //
    // The size of given axes may be fewer than the size of strides.
    // In that case, new strides will be composed by only given axes.
    //
    // It is the caller's responsibility to ensure validity of permutation.
    // If the permutation is invalid, the behavior is undefined.
    Strides Permute(const Axes& axes) const {
        CHAINERX_ASSERT(axes.size() <= size());
        Strides new_strides{};
        for (int8_t axe : axes) {
            new_strides.emplace_back(operator[](axe));
        }
        return new_strides;
    }
};

std::ostream& operator<<(std::ostream& os, const Strides& strides);

void CheckEqual(const Strides& lhs, const Strides& rhs);

// Returns a pair of lower and upper byte offsets to store the data.
// This forumula always holds: lower <= 0 < item_size <= upper
std::tuple<int64_t, int64_t> GetDataRange(const Shape& shape, const Strides& strides, size_t item_size);

}  // namespace chainerx
