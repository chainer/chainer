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
#include "xchainer/error.h"
#include "xchainer/ndim_vector.h"

namespace xchainer {

class Axes : public StackVector<int8_t, kMaxNdim> {
    using BaseVector = StackVector<int8_t, kMaxNdim>;

public:
    using const_iterator = BaseVector::const_iterator;
    using const_reverse_iterator = BaseVector::const_reverse_iterator;
    // TODO(niboshi): Declare other types required for this class to be a container.

    Axes() = default;

    // by iterators
    template <typename InputIt>
    Axes(InputIt first, InputIt last) {
        if (std::distance(first, last) > kMaxNdim) {
            throw DimensionError{"too many dimensions: ", std::distance(first, last)};
        }
        insert(begin(), first, last);
    }

    // by gsl:span
    explicit Axes(gsl::span<const int8_t> axes) : Axes{axes.begin(), axes.end()} {}

    // by initializer list
    Axes(std::initializer_list<int8_t> axes) : Axes{axes.begin(), axes.end()} {}

    // copy
    Axes(const Axes&) = default;
    Axes& operator=(const Axes&) = default;

    // move
    Axes(Axes&&) = default;
    Axes& operator=(Axes&&) = default;

    std::string ToString() const;

    int8_t ndim() const noexcept { return gsl::narrow_cast<int8_t>(size()); }

    const int8_t& operator[](int8_t index) const {
        if (!(0 <= index && static_cast<size_t>(index) < size())) {
            throw DimensionError{"index out of bounds"};
        }
        return this->StackVector::operator[](index);
    }

    // span
    gsl::span<const int8_t> span() const { return {*this}; }
};

std::ostream& operator<<(std::ostream&, const Axes&);

}  // namespace xchainer
