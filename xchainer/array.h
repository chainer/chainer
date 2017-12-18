#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include <gsl/gsl>

#include "xchainer/array_repr.h"
#include "xchainer/dtype.h"
#include "xchainer/shape.h"

namespace xchainer {

// The main data structure of multi-dimensional array.
class Array {
public:
    Array(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, int64_t offset = 0)
        : shape_(shape), is_contiguous_(true), dtype_(dtype), data_(std::move(data)), offset_(offset) {}

    Dtype dtype() const { return dtype_; }

    int8_t ndim() const { return shape_.ndim(); }

    const Shape& shape() const { return shape_; }

    bool is_contiguous() const { return is_contiguous_; }

    int64_t total_size() const { return shape_.total_size(); }

    int64_t element_bytes() const { return GetElementSize(dtype_); }

    int64_t total_bytes() const { return total_size() * element_bytes(); }

    const std::shared_ptr<void>& data() { return data_; }

    std::shared_ptr<const void> data() const { return data_; }

    int64_t offset() const { return offset_; }

    Array& iadd(const Array& other);
    Array& imul(const Array& other);
    Array add(const Array& other);
    Array mul(const Array& other);

private:
    Shape shape_;
    bool is_contiguous_;

    Dtype dtype_;

    std::shared_ptr<void> data_;
    int64_t offset_;

    template <typename T> Array& iadd(const Array& other);
    template <typename T> Array& imul(const Array& other);
    template <typename T> Array add(const Array& other);
    template <typename T> Array mul(const Array& other);
};

// Throws an exception if two arrays mismatch (for debug purpose)
void CheckEqual(const Array& lhs, const Array& rhs);

}  // namespace xchainer
