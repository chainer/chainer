#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array_body.h"
#include "xchainer/array_repr.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {

class Array;
class ArrayNode;

using ArrayRef = std::reference_wrapper<Array>;
using ConstArrayRef = std::reference_wrapper<const Array>;

namespace internal {

void SetUpOpNodes(const std::string& name, const std::vector<ConstArrayRef>& inputs, Array& out,
                  const std::vector<std::function<Array(const Array&, const std::vector<GraphId>&)>>& backward_functions,
                  const std::vector<GraphId>& graph_ids_to_stop_gradients = {});

bool HasArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

const std::shared_ptr<ArrayNode>& CreateArrayNode(Array& array, const GraphId& graph_id = kDefaultGraphId);

std::shared_ptr<const ArrayNode> GetArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

const std::shared_ptr<ArrayNode>& GetMutableArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

}  // namespace internal

enum class CopyKind {
    kCopy = 1,
    kView,
};

// The main data structure of multi-dimensional array.
class Array {
public:
    static Array FromBuffer(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device = GetDefaultDevice());

    static Array Empty(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
    static Array Full(const Shape& shape, Scalar scalar, Dtype dtype, Device& device = GetDefaultDevice());
    static Array Full(const Shape& shape, Scalar scalar, Device& device = GetDefaultDevice());
    static Array Zeros(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
    static Array Ones(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());

    // Creates an array which has the same shape and dtype as the other array.
    // The new array is allocated in the default device. The device of the other array
    // is ignored.
    static Array EmptyLike(const Array& array, Device& device = GetDefaultDevice());
    static Array FullLike(const Array& array, Scalar scalar, Device& device = GetDefaultDevice());
    static Array ZerosLike(const Array& array, Device& device = GetDefaultDevice());
    static Array OnesLike(const Array& array, Device& device = GetDefaultDevice());

    explicit Array(gsl::not_null<std::shared_ptr<internal::ArrayBody>> body) : body_(std::move(body)) {}

    // Deep copy ctor and copy assignment
    Array(const Array& other);
    Array(Array&& other) = default;

    Array& operator=(const Array&) = delete;
    Array& operator=(Array&& other) = delete;

    Array& operator+=(const Array& rhs);
    Array& operator*=(const Array& rhs);
    Array operator+(const Array& rhs) const;
    Array operator*(const Array& rhs) const;

    // Returns a transposed view of the array.
    Array Transpose() const;

    // Creates a copy. It will be connected to all the graphs.
    Array Copy() const;

    // Transfers the array to another device. It will be connected to all the graphs.
    //
    // If the destination is the same device, an array with aliased data is returned.
    // Otherwise, a C-contiguous Array will be created on the target device.
    // TODO(niboshi): Currently control over whether to make an alias is not supported.
    Array ToDevice(Device& dst_device) const;

    // Creates a copy or a view. It will be disconnected from all the graphs.
    Array AsConstant(CopyKind kind = CopyKind::kView) const;

    // Creates a copy or a view. It will be disconnected from the specified graphs.
    Array AsConstant(const std::vector<GraphId>& graph_ids, CopyKind kind = CopyKind::kView) const;

    void Fill(Scalar value);

    const nonstd::optional<Array>& GetGrad(const GraphId& graph_id = kDefaultGraphId) const;

    void SetGrad(Array grad, const GraphId& graph_id = kDefaultGraphId);

    // Clears the gradient stored in the ArrayNode, but does not delete the ArrayNode itself
    void ClearGrad(const GraphId& graph_id = kDefaultGraphId);

    bool IsGradRequired(const GraphId& graph_id = kDefaultGraphId) const { return internal::HasArrayNode(*this, graph_id); }

    // Creates a new ArrayNode to store the gradient
    Array& RequireGrad(const GraphId& graph_id = kDefaultGraphId) {
        internal::CreateArrayNode(*this, graph_id);
        return *this;
    }

    int64_t GetTotalSize() const { return shape().GetTotalSize(); }

    int64_t GetTotalBytes() const { return GetTotalSize() * element_bytes(); }

    bool IsContiguous() const { return internal::IsContiguous(shape(), strides(), element_bytes()); }

    std::string ToString() const;

    const std::shared_ptr<internal::ArrayBody>& body() { return body_; }

    std::shared_ptr<const internal::ArrayBody> body() const { return body_; }

    std::shared_ptr<internal::ArrayBody>&& move_body() { return std::move(body_); }

    Dtype dtype() const { return body_->dtype_; }

    Device& device() const { return body_->device_; }

    int8_t ndim() const { return shape().ndim(); }

    const Shape& shape() const { return body_->shape_; }

    const Strides& strides() const { return body_->strides_; }

    int64_t element_bytes() const { return GetElementSize(dtype()); }

    const std::shared_ptr<void>& data() { return body_->data_; }

    std::shared_ptr<const void> data() const { return body_->data_; }

    void* raw_data() { return body_->data_.get(); }

    const void* raw_data() const { return body_->data_.get(); }

    int64_t offset() const { return body_->offset_; }

    const std::vector<std::shared_ptr<ArrayNode>>& nodes() const { return body_->nodes_; };

    std::vector<std::shared_ptr<ArrayNode>>& nodes() { return body_->nodes_; };

private:
    Array(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset = 0);

    void Add(const Array& rhs, Array& out) const;
    void Mul(const Array& rhs, Array& out) const;

    std::shared_ptr<internal::ArrayBody> body_;
};

void DebugDumpComputationalGraph(std::ostream& os, const Array& array, const GraphId& graph_id, int indent = 0);

}  // namespace xchainer
