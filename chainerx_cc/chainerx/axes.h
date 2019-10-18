#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <string>

#include <absl/types/span.h>
#include <gsl/gsl>

#include "chainerx/constant.h"
#include "chainerx/error.h"
#include "chainerx/optional_container_arg.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

class Axes : public StackVector<int8_t, kMaxNdim> {
    using BaseVector = StackVector<int8_t, kMaxNdim>;

public:
    using const_iterator = BaseVector::const_iterator;
    using const_reverse_iterator = BaseVector::const_reverse_iterator;
    // TODO(niboshi): Declare other types required for this class to be a container.

    Axes() = default;

    ~Axes() = default;

    // by iterators
    template <typename InputIt>
    Axes(InputIt first, InputIt last) {
        if (std::distance(first, last) > kMaxNdim) {
            throw DimensionError{"too many dimensions: ", std::distance(first, last)};
        }
        insert(begin(), first, last);
    }

    // by span
    explicit Axes(absl::Span<const int8_t> axes) : Axes{axes.begin(), axes.end()} {}

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

    int8_t& operator[](int8_t index) {
        if (!(0 <= index && static_cast<size_t>(index) < size())) {
            throw DimensionError{"index out of bounds"};
        }
        return this->StackVector::operator[](index);
    }

    const int8_t& operator[](int8_t index) const {
        if (!(0 <= index && static_cast<size_t>(index) < size())) {
            throw DimensionError{"index out of bounds"};
        }
        return this->StackVector::operator[](index);
    }

    // span
    absl::Span<const int8_t> span() const { return {*this}; }
};

std::ostream& operator<<(std::ostream& os, const Axes& axes);

using OptionalAxes = OptionalContainerArg<Axes>;

namespace internal {

bool IsAxesPermutation(const Axes& axes, int8_t ndim);

// Normalizes possibly-negative axis to non-negative axis in [0, ndim).
// If `axis` does not fit in [-ndim, ndim), DimensionError is thrown.
int8_t NormalizeAxis(int8_t axis, int8_t ndim);

// Resolves the axis argument of many operations.
// Negative axes are converted to non-negative ones (by wrapping at ndim).
Axes GetNormalizedAxes(const Axes& axis, int8_t ndim);

// Resolves the axis argument of many operations.
// Negative axes are converted to non-negative ones (by wrapping at ndim).
// Axes are then sorted.
Axes GetSortedAxes(const Axes& axis, int8_t ndim);

// Resolves the axis argument of many operations.
// Negative axes are converted to non-negative ones (by wrapping at ndim).
// Axes are then sorted.
// nullopt is converted to a vector of all axes.
Axes GetSortedAxesOrAll(const OptionalAxes& axis, int8_t ndim);

}  // namespace internal
}  // namespace chainerx
