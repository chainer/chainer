#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array_body.h"
#include "xchainer/array_index.h"
#include "xchainer/array_repr.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/enum.h"
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

Array MakeArray(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset = 0);

void SetUpOpNodes(
        const std::string& name,
        const std::vector<ConstArrayRef>& inputs,
        const Array& out,
        const std::vector<std::function<Array(const Array&, const std::vector<GraphId>&)>>& backward_functions,
        const std::vector<GraphId>& graph_ids_to_stop_gradients = {});

bool HasArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

const std::shared_ptr<ArrayNode>& CreateArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

std::shared_ptr<const ArrayNode> GetArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

const std::shared_ptr<ArrayNode>& GetMutableArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

}  // namespace internal

// The main data structure of multi-dimensional array.
class Array {
public:
    static Array FromContiguousHostData(
            const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device = GetDefaultDevice());
    static Array Empty(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
    static Array Full(const Shape& shape, Scalar fill_value, Dtype dtype, Device& device = GetDefaultDevice());
    static Array Full(const Shape& shape, Scalar fill_value, Device& device = GetDefaultDevice());
    static Array Zeros(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
    static Array Ones(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());

    // Creates an array which has the same shape and dtype as the other array.
    // The new array is allocated in the default device. The device of the other array
    // is ignored.
    static Array EmptyLike(const Array& a, Device& device = GetDefaultDevice());
    static Array FullLike(const Array& a, Scalar fill_value, Device& device = GetDefaultDevice());
    static Array ZerosLike(const Array& a, Device& device = GetDefaultDevice());
    static Array OnesLike(const Array& a, Device& device = GetDefaultDevice());

    Array() = default;

    explicit Array(gsl::not_null<std::shared_ptr<internal::ArrayBody>> body) : body_(std::move(body)) {}

    // Copy constructor that copies the pointer to the body instead of the body itself.
    //
    // Use MakeView if you want to clone the body.
    Array(const Array& other) = default;
    Array(Array&& other) = default;

    // Assign operators just replace the body. They do not copy data between arrays.
    Array& operator=(const Array&) = default;
    Array& operator=(Array&& other) = default;

    Array operator-() const;

    Array operator==(const Array& rhs) const;

    Array& operator+=(const Array& rhs);
    const Array& operator+=(const Array& rhs) const;
    Array& operator-=(const Array& rhs);
    const Array& operator-=(const Array& rhs) const;
    Array& operator*=(const Array& rhs);
    const Array& operator*=(const Array& rhs) const;
    Array operator+(const Array& rhs) const;
    Array operator-(const Array& rhs) const;
    Array operator*(const Array& rhs) const;
    Array operator*(Scalar rhs) const;

    // Returns a view selected with the indices.
    Array At(const std::vector<ArrayIndex>& indices) const;

    // Returns a transposed view of the array.
    Array Transpose() const;

    // Returns a reshaped array.
    // TODO(niboshi): Support reshape that require a copy.
    // TODO(niboshi): Support shape with dimension -1.
    Array Reshape(const Shape& newshape) const;

    // Returns a squeezed array with unit-length axes removed.
    //
    // If no axes are specified, all axes of unit-lengths are removed.
    // If no axes can be removed, an array with aliased data is returned.
    Array Squeeze(const nonstd::optional<std::vector<int8_t>>& axis = nonstd::nullopt) const;

    // Broadcasts the array to the specified shape.
    // Returned array is always a view to this array.
    Array BroadcastTo(const Shape& shape) const;

    // Returns the indices of the maximum values along the given axis.
    Array ArgMax(const nonstd::optional<int8_t>& axis = nonstd::nullopt) const;

    // Returns a sum of the array.
    // If `axis` is set, it will be summed over the specified axes.
    // Otherwise, it will be summed over all the existing axes.
    // Note: When implementing xchainer::Sum(), be careful of the semantics of the default value of `keepdims`. See NumPy documentation.
    Array Sum(const nonstd::optional<std::vector<int8_t>>& axis = nonstd::nullopt, bool keepdims = false) const;

    // Returns a dot product of the array with another one.
    Array Dot(const Array& b) const;

    // Creates a copy.
    // It will be connected to all the graphs.
    // It will be always C-contiguous.
    Array Copy() const;

    // Creates a view.
    // It does not make a new node for any graphs.
    Array MakeView() const;

    // Transfers the array to another device. It will be connected to all the graphs.
    //
    // If the destination is the same device, an array with aliased data is returned.
    // Otherwise, a C-contiguous Array will be created on the target device.
    // TODO(niboshi): Currently control over whether to make an alias is not supported.
    Array ToDevice(Device& dst_device) const;

    // Transfer the array to the native device. It will be connected to all the graphs.
    //
    // This is a wrapper function which calls Array::ToDevice with the native:0 device.
    // See also: Array::ToDevice();
    Array ToNative() const;

    // Creates a copy or a view. It will be disconnected from all the graphs.
    // If `kind` is `CopyKind::kCopy`, the returned array will be always C-contiguous.
    Array AsConstant(CopyKind kind = CopyKind::kView) const;

    // Creates a copy or a view. It will be disconnected from the specified graphs.
    // If `kind` is `CopyKind::kCopy`, the returned array will be always C-contiguous.
    Array AsConstant(const std::vector<GraphId>& graph_ids, CopyKind kind = CopyKind::kView) const;

    void Fill(Scalar value) const;

    const nonstd::optional<Array>& GetGrad(const GraphId& graph_id = kDefaultGraphId) const;

    void SetGrad(Array grad, const GraphId& graph_id = kDefaultGraphId) const;

    // Clears the gradient stored in the ArrayNode, but does not delete the ArrayNode itself
    void ClearGrad(const GraphId& graph_id = kDefaultGraphId) const;

    bool IsGradRequired(const GraphId& graph_id = kDefaultGraphId) const { return internal::HasArrayNode(*this, graph_id); }

    // Creates a new ArrayNode to store the gradient
    const Array& RequireGrad(const GraphId& graph_id = kDefaultGraphId) const {
        internal::CreateArrayNode(*this, graph_id);
        return *this;
    }

    // Creates a new ArrayNode to store the gradient
    Array& RequireGrad(const GraphId& graph_id = kDefaultGraphId) {
        internal::CreateArrayNode(*this, graph_id);
        return *this;
    }

    int64_t GetTotalSize() const { return shape().GetTotalSize(); }

    int64_t GetTotalBytes() const { return GetTotalSize() * element_bytes(); }

    // Returns the effective contiguous memory address space occupied by this array.
    // The last element in the span refers to the past-the-end array element.
    gsl::span<const uint8_t> GetDataRange() const;

    bool IsContiguous() const { return internal::IsContiguous(shape(), strides(), element_bytes()); }

    std::string ToString() const;

    const std::shared_ptr<internal::ArrayBody>& body() const { return body_; }

    std::shared_ptr<internal::ArrayBody>&& move_body() { return std::move(body_); }

    Dtype dtype() const { return body_->dtype_; }

    Device& device() const { return body_->device_; }

    int8_t ndim() const { return shape().ndim(); }

    const Shape& shape() const { return body_->shape_; }

    const Strides& strides() const { return body_->strides_; }

    int64_t element_bytes() const { return GetElementSize(dtype()); }

    const std::shared_ptr<void>& data() const { return body_->data_; }

    void* raw_data() const { return body_->data_.get(); }

    int64_t offset() const { return body_->offset_; }

    std::vector<std::shared_ptr<ArrayNode>>& nodes() const { return body_->nodes_; }

private:
    friend Array internal::MakeArray(
            const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

    Array(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset = 0);

    std::shared_ptr<internal::ArrayBody> body_;
};

inline Array operator*(Scalar lhs, const Array& rhs) { return rhs * lhs; }

void DebugDumpComputationalGraph(std::ostream& os, const Array& array, const GraphId& graph_id, int indent = 0);

}  // namespace xchainer
