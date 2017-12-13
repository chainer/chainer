#pragma once

#include <array>
#include <cstdint>
#include <gsl/gsl>
#include <initializer_list>
#include "xchainer/constant.h"
#include "xchainer/error.h"

namespace xchainer {

class Shape {
    using DimsType = std::array<int64_t, kMaxNdim>;

public:
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
    Shape(Shape&&) = default;
    Shape& operator=(Shape&&) = delete;

    size_t size() const noexcept { return static_cast<size_t>(ndim_); }

    int8_t ndim() const noexcept { return ndim_; }

    int64_t total_size() const;

    int64_t operator[](int8_t index) const noexcept {
        Expects(0 <= index && index < ndim_);
        return dims_[index];
    }

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

}  // namespace xchainer
