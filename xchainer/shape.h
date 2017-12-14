#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>

#include <gsl/gsl>

#include "xchainer/constant.h"
#include "xchainer/error.h"

namespace xchainer {

class Shape {
    using DimsType = std::array<int64_t, kMaxNdim>;

public:
    using const_iterator = DimsType::const_iterator;
    using const_reverse_iterator = DimsType::const_reverse_iterator;

    // by gsl:span
    Shape(gsl::span<const int64_t> dims) : ndim_(gsl::narrow_cast<int8_t>(dims.size())) {
        CheckNdim();
        std::copy(dims.begin(), dims.end(), dims_.begin());
    }

    // by initializer list
    Shape(std::initializer_list<int64_t> dims) : Shape(gsl::make_span(dims.begin(), dims.end())) {}

    // copy
    Shape(const Shape&) = default;
    Shape& operator=(const Shape&) = delete;

    // move
    Shape(Shape&&) = delete;
    Shape& operator=(Shape&&) = delete;

    size_t size() const noexcept { return static_cast<size_t>(ndim_); }

    int8_t ndim() const noexcept { return ndim_; }

    int64_t total_size() const;

    int64_t operator[](int8_t index) const {
        if (!(0 <= index && index < ndim_)) {
            throw DimensionError("index out of bounds");
        }
        return dims_[index];
    }

    // iterators
    const_iterator cbegin() const noexcept { return dims_.cbegin(); }
    const_iterator cend() const noexcept { return dims_.cbegin() + ndim_; }
    const_iterator begin() const noexcept { return cbegin(); }
    const_iterator end() const noexcept { return cend(); }

    // reverse iterators
    const_reverse_iterator crbegin() const noexcept { return dims_.crend() - ndim_; }
    const_reverse_iterator crend() const noexcept { return dims_.crend(); }
    const_reverse_iterator rbegin() const noexcept { return crbegin(); }
    const_reverse_iterator rend() const noexcept { return crend(); }

    // span
    gsl::span<const int64_t> span() const { return {&dims_[0], static_cast<size_t>(ndim_)}; }

private:
    void CheckNdim() const {
        if (ndim_ > kMaxNdim) {
            throw DimensionError("too many dimensions");
        }
    }

    DimsType dims_;
    int8_t ndim_;
};

inline bool operator==(const Shape& lhs, const Shape& rhs) { return lhs.span() == rhs.span(); }

inline bool operator!=(const Shape& lhs, const Shape& rhs) { return lhs.span() != rhs.span(); }

void CheckEqual(const Shape& lhs, const Shape& rhs);

}  // namespace xchainer
