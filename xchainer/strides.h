#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <string>

#include <gsl/gsl>

#include "xchainer/constant.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/shape.h"

namespace xchainer {

class Strides : public StackVector<int64_t, kMaxNdim> {
    using BaseVector = StackVector<int64_t, kMaxNdim>;

public:
    using const_iterator = BaseVector::const_iterator;
    using const_reverse_iterator = BaseVector::const_reverse_iterator;

    Strides() = default;

    // Creates strides for contiguous array.
    Strides(const Shape& shape, Dtype dtype) : Strides{shape, GetElementSize(dtype)} {}
    Strides(const Shape& shape, int64_t element_size);

    // by iterators
    template <typename InputIt>
    Strides(InputIt first, InputIt last) {
        if (std::distance(first, last) > kMaxNdim) {
            throw DimensionError{"too many dimensions: ", std::distance(first, last)};
        }
        insert(begin(), first, last);
    }

    // by gsl:span
    explicit Strides(gsl::span<const int64_t> dims) : Strides{dims.begin(), dims.end()} {}

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
            throw DimensionError{"index out of bounds"};
        }
        return this->StackVector::operator[](index);
    }

    // span
    gsl::span<const int64_t> span() const { return {*this}; }
};

std::ostream& operator<<(std::ostream&, const Strides&);

void CheckEqual(const Strides& lhs, const Strides& rhs);

}  // namespace xchainer
