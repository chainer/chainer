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

class Strides;

class Shape : public StackVector<int64_t, kMaxNdim> {
    using BaseVector = StackVector<int64_t, kMaxNdim>;

public:
    using const_iterator = BaseVector::const_iterator;
    using const_reverse_iterator = BaseVector::const_reverse_iterator;

    Shape() {}

    // by iterators
    template <typename InputIt>
    Shape(InputIt first, InputIt last) {
        CheckNdim(std::distance(first, last));
        insert(begin(), first, last);
    }

    // by gsl:span
    explicit Shape(gsl::span<const int64_t> dims) : Shape{dims.begin(), dims.end()} {}

    // by initializer list
    Shape(std::initializer_list<int64_t> dims) : Shape{dims.begin(), dims.end()} {}

    // copy
    Shape(const Shape& other) = default;
    Shape& operator=(const Shape&) = default;

    // move
    Shape(Shape&&) = default;
    Shape& operator=(Shape&&) = default;

    int64_t GetTotalSize() const;

    std::string ToString() const;

    int8_t ndim() const noexcept { return gsl::narrow_cast<int8_t>(size()); }

    const int64_t& operator[](int8_t index) const {
        if (!(0 <= index && static_cast<size_t>(index) < size())) {
            throw DimensionError("index out of bounds");
        }
        return this->StackVector::operator[](index);
    }

    // span
    gsl::span<const int64_t> span() const { return {*this}; }

private:
    void CheckNdim(std::ptrdiff_t ndim) const {
        if (ndim > kMaxNdim) {
            throw DimensionError("too many dimensions: " + std::to_string(ndim));
        }
    }
};

namespace internal {

bool IsContiguous(const Shape& shape, const Strides& strides, int64_t element_bytes);

Shape BroadcastShapes(const Shape& shape0, const Shape& shape1);

bool IsValidReductionShape(const Shape& in_shape, const NdimVector<int8_t>& axis, const Shape& out_shape, bool allow_keepdims);

Shape TransposeShape(const Shape& shape, const NdimVector<int8_t>& axes);

}  // namespace internal

inline bool operator==(const Shape& lhs, const Shape& rhs) { return lhs.span() == rhs.span(); }

inline bool operator!=(const Shape& lhs, const Shape& rhs) { return lhs.span() != rhs.span(); }

std::ostream& operator<<(std::ostream&, const Shape&);

void CheckEqual(const Shape& lhs, const Shape& rhs);

}  // namespace xchainer
