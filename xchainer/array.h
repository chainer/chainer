#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include <gsl/gsl>

#include "xchainer/array_node.h"
#include "xchainer/array_repr.h"
#include "xchainer/dtype.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {

// The main data structure of multi-dimensional array.
class Array {
public:
    Array(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, bool requires_grad = false, int64_t offset = 0);

    // Copy assignment operator is deleted to avoid performance drops due to possible unwanted copies and heavy refactorings later on until
    // the behavior is better agreed upon
    Array& operator=(const Array&) = delete;

    static Array Empty(const Shape& shape, Dtype dtype);
    static Array Full(const Shape& shape, Dtype dtype, const Scalar& scalar);
    static Array Full(const Shape& shape, const Scalar& scalar);
    static Array Zeros(const Shape& shape, Dtype dtype);
    static Array Ones(const Shape& shape, Dtype dtype);

    // Creates an array which has the same shape and dtype as the other array.
    // The new array is allocated in the current device. The device of the other array
    // is ignored.
    static Array EmptyLike(const Array& array);
    static Array FullLike(const Array& array, const Scalar& scalar);
    static Array ZerosLike(const Array& array);
    static Array OnesLike(const Array& array);

    Dtype dtype() const { return dtype_; }

    int8_t ndim() const { return shape_.ndim(); }

    const Shape& shape() const { return shape_; }

    bool is_contiguous() const { return is_contiguous_; }

    int64_t total_size() const { return shape_.total_size(); }

    int64_t element_bytes() const { return GetElementSize(dtype_); }

    int64_t total_bytes() const { return total_size() * element_bytes(); }

    const std::shared_ptr<void>& data() { return data_; }

    std::shared_ptr<const void> data() const { return data_; }

    bool requires_grad() const { return requires_grad_; }

    int64_t offset() const { return offset_; }

    const std::shared_ptr<ArrayNode>& node() { return node_; }

    std::shared_ptr<const ArrayNode> node() const { return node_; }

    const std::shared_ptr<ArrayNode>& RenewNode() {
        node_ = std::make_shared<ArrayNode>();
        return node_;
    }

    Array& operator+=(const Array& rhs);
    Array& operator*=(const Array& rhs);
    Array operator+(const Array& rhs) const;
    Array operator*(const Array& rhs) const;

    void Fill(Scalar value);

    std::string ToString() const;

    Array DeepCopy() const;

private:
    void Add(const Array& rhs, Array& out) const;
    void Mul(const Array& rhs, Array& out) const;

    Shape shape_;
    bool is_contiguous_;

    Dtype dtype_;

    std::shared_ptr<void> data_;
    bool requires_grad_;
    int64_t offset_;

    std::shared_ptr<ArrayNode> node_;
};

void DebugDumpComputationalGraph(std::ostream& os, const Array& array, int indent = 0);

}  // namespace xchainer
