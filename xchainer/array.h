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
#include "xchainer/array_node.h"
#include "xchainer/array_repr.h"
#include "xchainer/axes.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/enum.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {

class Array;

using ArrayRef = std::reference_wrapper<Array>;
using ConstArrayRef = std::reference_wrapper<const Array>;

namespace internal {

Array MakeArray(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset = 0);

bool HasArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

bool HasAnyArrayNode(const Array& array);

// Creates a new array node on the specified graph.
// XchainerError is thrown if an array node is already registered on the graph.
// The returned reference is only valid until the next call of CreateArrayNode (or ArrayBody::AddNode) on the same ArrayBody
// instance.
const std::shared_ptr<ArrayNode>& CreateArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

std::shared_ptr<const ArrayNode> GetArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

const std::shared_ptr<ArrayNode>& GetMutableArrayNode(const Array& array, const GraphId& graph_id = kDefaultGraphId);

}  // namespace internal

// The main data structure of multi-dimensional array.
class Array {
public:
    Array() = default;

    explicit Array(gsl::not_null<std::shared_ptr<internal::ArrayBody>> body) : body_{std::move(body)} {}

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
    Array& operator+=(Scalar rhs);
    Array& operator-=(const Array& rhs);
    Array& operator-=(Scalar rhs);
    Array& operator*=(const Array& rhs);
    Array& operator*=(Scalar rhs);
    Array& operator/=(const Array& rhs);
    Array& operator/=(Scalar rhs);

    const Array& operator+=(const Array& rhs) const;
    const Array& operator+=(Scalar rhs) const;
    const Array& operator-=(const Array& rhs) const;
    const Array& operator-=(Scalar rhs) const;
    const Array& operator*=(const Array& rhs) const;
    const Array& operator*=(Scalar rhs) const;
    const Array& operator/=(const Array& rhs) const;
    const Array& operator/=(Scalar rhs) const;

    Array operator+(const Array& rhs) const;
    Array operator+(Scalar rhs) const;
    Array operator-(const Array& rhs) const;
    Array operator-(Scalar rhs) const;
    Array operator*(const Array& rhs) const;
    Array operator*(Scalar rhs) const;
    Array operator/(const Array& rhs) const;
    Array operator/(Scalar rhs) const;

    // Returns a view selected with the indices.
    Array At(const std::vector<ArrayIndex>& indices) const;

    // Returns a transposed view of the array.
    Array Transpose(const OptionalAxes& axes = nonstd::nullopt) const;

    // Returns a reshaped array.
    // TODO(niboshi): Support shape with dimension -1.
    Array Reshape(const Shape& newshape) const;

    // Returns a squeezed array with unit-length axes removed.
    //
    // If no axes are specified, all axes of unit-lengths are removed.
    // If no axes can be removed, an array with aliased data is returned.
    Array Squeeze(const OptionalAxes& axis = nonstd::nullopt) const;

    // Broadcasts the array to the specified shape.
    // Returned array is always a view to this array.
    Array BroadcastTo(const Shape& shape) const;

    // Returns the indices of the maximum values along the given axis.
    Array ArgMax(const OptionalAxes& axis = nonstd::nullopt) const;

    // Returns a sum of the array.
    // If `axis` is set, it will be summed over the specified axes.
    // Otherwise, it will be summed over all the existing axes.
    // Note: When implementing xchainer::Sum(), be careful of the semantics of the default value of `keepdims`. See NumPy documentation.
    Array Sum(const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false) const;

    // Returns the maximum value of the array.
    // If `axis` is set, the maximum value is chosen along the specified axes.
    // Otherwise, all the elements are searched at once.
    Array Max(const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false) const;

    // Returns a dot product of the array with another one.
    Array Dot(const Array& b) const;

    // Takes elements specified by indices from the array.
    //
    // TODO(niboshi): Support Scalar and StackVector as indices.
    // TODO(niboshi): Support axis=None behavior in NumPy.
    // TODO(niboshi): Support indices dtype other than int64.
    Array Take(const Array& indices, int8_t axis) const;

    // Creates a copy.
    // It will be connected to all the graphs.
    // It will be always C-contiguous.
    Array Copy() const;

    // Creates a view.
    // It creates a new array node and connects graphs.
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
    Array AsGradStopped(CopyKind kind = CopyKind::kView) const;

    // Creates a copy or a view. It will be disconnected from the specified graphs.
    // If `kind` is `CopyKind::kCopy`, the returned array will be always C-contiguous.
    Array AsGradStopped(gsl::span<const GraphId> graph_ids, CopyKind kind = CopyKind::kView) const;
    Array AsGradStopped(std::initializer_list<const GraphId> graph_ids, CopyKind kind = CopyKind::kView) const {
        return AsGradStopped(gsl::span<const GraphId>{graph_ids.begin(), graph_ids.end()}, kind);
    }

    // Casts to a specified type.
    // By default, always returns a newly allocated array. If `copy` is false,
    // and the dtype requirement is satisfied, the input array is returned instead of a copy.
    Array AsType(Dtype dtype, bool copy = true) const;

    void Fill(Scalar value) const;

    const nonstd::optional<Array>& GetGrad(const GraphId& graph_id = kDefaultGraphId) const;

    void SetGrad(Array grad, const GraphId& graph_id = kDefaultGraphId) const;

    // Clears the gradient stored in the ArrayNode, but does not delete the ArrayNode itself
    void ClearGrad(const GraphId& graph_id = kDefaultGraphId) const;

    // Returns whether the array needs to backprop.
    // This takes into account NoBackpropModeScope and ForceBackpropModeScope.
    bool IsGradRequired(const GraphId& graph_id = kDefaultGraphId) const;
    bool IsGradRequired(AnyGraph any_graph) const;

    // Flags the array to compute the gradient during backprop.
    // If the backprop mode is disabled for the graph in the current thread, it does nothing but returns a reference to itself.
    const Array& RequireGrad(const GraphId& graph_id = kDefaultGraphId) const { return RequireGradImpl(*this, graph_id); }

    Array& RequireGrad(const GraphId& graph_id = kDefaultGraphId) { return RequireGradImpl(*this, graph_id); }

    int64_t GetTotalSize() const { return shape().GetTotalSize(); }

    int64_t GetNBytes() const { return GetTotalSize() * item_size(); }

    bool IsContiguous() const { return internal::IsContiguous(shape(), strides(), item_size()); }

    std::string ToString() const;

    const std::shared_ptr<internal::ArrayBody>& body() const { return body_; }

    std::shared_ptr<internal::ArrayBody>&& move_body() { return std::move(body_); }

    Dtype dtype() const { return body_->dtype_; }

    Device& device() const { return body_->device_; }

    int8_t ndim() const { return shape().ndim(); }

    const Shape& shape() const { return body_->shape_; }

    const Strides& strides() const { return body_->strides_; }

    int64_t item_size() const { return GetItemSize(dtype()); }

    const std::shared_ptr<void>& data() const { return body_->data_; }

    void* raw_data() const { return body_->data_.get(); }

    int64_t offset() const { return body_->offset_; }

    std::vector<std::shared_ptr<ArrayNode>>& nodes() const { return body_->nodes_; }

private:
    friend Array internal::MakeArray(
            const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

    Array(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset = 0);

    template <typename T>
    static T& RequireGradImpl(T& array, const GraphId& graph_id);

    std::shared_ptr<internal::ArrayBody> body_;
};

inline Array operator+(Scalar lhs, const Array& rhs) { return rhs + lhs; }
inline Array operator-(Scalar lhs, const Array& rhs) { return -rhs + lhs; }
inline Array operator*(Scalar lhs, const Array& rhs) { return rhs * lhs; }
// TODO(hvy): Implement Scalar / Array using e.g. multiplication with reciprocal.

void DebugDumpComputationalGraph(
        std::ostream& os,
        const Array& array,
        const GraphId& graph_id,
        int indent = 0,
        const std::vector<std::pair<ConstArrayRef, std::string>>& array_name_map = {});

}  // namespace xchainer
