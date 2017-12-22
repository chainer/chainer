#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include <gsl/gsl>

#include "xchainer/array_node.h"
#include "xchainer/dtype.h"
#include "xchainer/shape.h"

namespace xchainer {

// The main data structure of multi-dimensional array.
class Array {
public:
    Array(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, int64_t offset = 0);

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

    const std::shared_ptr<ArrayNode>& node() { return node_; }

    std::shared_ptr<const ArrayNode> node() const { return node_; }

    const std::shared_ptr<ArrayNode>& CreateNewNode() {
        node_ = std::make_shared<ArrayNode>();
        return node_;
    }

    Array& IAdd(const Array& rhs);
    Array& IMul(const Array& rhs);
    Array Add(const Array& rhs) const;
    Array Mul(const Array& rhs) const;

    std::string ToString() const;

private:
    template <typename T>
    void Add(const Array& rhs, Array& out) const;
    template <typename T>
    void Mul(const Array& rhs, Array& out) const;

    void Add(const Array& rhs, Array& out) const;
    void Mul(const Array& rhs, Array& out) const;

    Shape shape_;
    bool is_contiguous_;

    Dtype dtype_;

    std::shared_ptr<void> data_;
    int64_t offset_;

    std::shared_ptr<ArrayNode> node_;
};

void DebugDumpComputationalGraph(std::ostream& os, const Array& array, int indent = 0);

}  // namespace xchainer
