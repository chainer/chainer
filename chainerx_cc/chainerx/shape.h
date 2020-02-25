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

#include "chainerx/axes.h"
#include "chainerx/constant.h"
#include "chainerx/dims.h"
#include "chainerx/error.h"

namespace chainerx {

class Strides;

class Shape : public Dims {
    using BaseVector = Dims;

public:
    using const_iterator = BaseVector::const_iterator;
    using const_reverse_iterator = BaseVector::const_reverse_iterator;
    // TODO(niboshi): Declare other types required for this class to be a container.

    Shape() = default;

    ~Shape() = default;

    // by iterators
    template <typename InputIt>
    Shape(InputIt first, InputIt last) {
        if (std::distance(first, last) > kMaxNdim) {
            throw DimensionError{"too many dimensions: ", std::distance(first, last)};
        }
        insert(begin(), first, last);
    }

    // by span
    explicit Shape(absl::Span<const int64_t> dims) : Shape{dims.begin(), dims.end()} {}

    // by initializer list
    Shape(std::initializer_list<int64_t> dims) : Shape{dims.begin(), dims.end()} {}

    // copy
    Shape(const Shape&) = default;
    Shape& operator=(const Shape&) = default;

    // move
    Shape(Shape&&) = default;
    Shape& operator=(Shape&&) = default;

    int64_t GetTotalSize() const;

    std::string ToString() const;

    int8_t ndim() const noexcept { return gsl::narrow_cast<int8_t>(size()); }

    const int64_t& operator[](int8_t index) const {
        if (!(0 <= index && static_cast<size_t>(index) < size())) {
            throw IndexError{"Shape index ", index, " out of bounds for shape with ", size(), " size."};
        }
        return this->Dims::operator[](index);
    }

    int64_t& operator[](int8_t index) {
        if (!(0 <= index && static_cast<size_t>(index) < size())) {
            throw IndexError{"Shape index ", index, " out of bounds for shape with ", size(), " size."};
        }
        return this->Dims::operator[](index);
    }

    // span
    absl::Span<const int64_t> span() const { return {*this}; }

    Shape Permute(const Axes& axes) const {
        CHAINERX_ASSERT(axes.size() <= size());
        Shape new_shape{};
        for (int8_t axe : axes) {
            new_shape.emplace_back(operator[](axe));
        }
        return new_shape;
    }
};

namespace internal {

bool IsContiguous(const Shape& shape, const Strides& strides, int64_t item_size);

// Returns true if a reduction can take place under the given conditions, only considering the number of dimensions.
// Otherwise, returns false.
//
// TODO(hvy): Check the dimension lengths too and reconsider the interface. E.g. return void and assert inside the function if only used for
// assertions.
bool IsValidReductionShape(const Shape& in_shape, const Axes& axes, const Shape& out_shape, bool allow_keepdims);

int64_t CountItemsAlongAxes(const Shape& shape, const Axes& axes);

Shape BroadcastShapes(const Shape& shape0, const Shape& shape1);

// Returns a shape where axes are reduced.
Shape ReduceShape(const Shape& shape, const Axes& axes, bool keepdims);

// Returns a shape with additional axes, with length 1.
Shape ExpandShape(const Shape& shape, const Axes& axes);

Shape TransposeShape(const Shape& shape, const Axes& axes);

}  // namespace internal

std::ostream& operator<<(std::ostream& os, const Shape& shape);

void CheckEqual(const Shape& lhs, const Shape& rhs);

}  // namespace chainerx
