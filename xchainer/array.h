#pragma once

#include <cstdint>
#include <gsl/gsl>
#include <memory>
#include <utility>
#include "xchainer/dtype.h"
#include "xchainer/shape.h"

namespace xchainer {

// The main data structure of multi-dimensional array.
class Array {
public:
    Array(const Shape& shape, Dtype dtype);

    Dtype dtype() const { return dtype_; }

    int8_t ndim() const { return shape_.ndim(); }

    const Shape& shape() const { return shape_; }

    bool is_contiguous() const { return is_contiguous_; }

    int64_t total_size() const { return shape_.total_size(); }

    int64_t element_bytes() const { return GetElementSize(dtype_); }

    int64_t total_bytes() const { return total_size() * element_bytes(); }

    const std::shared_ptr<void>& data() const { return data_; }

    int64_t offset() const { return offset_; }

    void SetContiguousData(std::shared_ptr<void> data, int64_t offset = 0);

    void SetContiguousData(Array& other, int64_t relative_offset = 0) { SetContiguousData(other.data_, other.offset_ + relative_offset); }

    std::shared_ptr<Array> MakeSimilar() const { return std::make_shared<Array>(shape_, dtype_); }

private:
    Shape shape_;
    bool is_contiguous_;

    Dtype dtype_;

    std::shared_ptr<void> data_;
    int64_t offset_;
};

}  // namespace xchainer
