#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array_repr.h"
#include "xchainer/constant.h"
#include "xchainer/device_id.h"
#include "xchainer/dtype.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {

class Array;
class ArrayNode;

using GraphId = std::string;

namespace internal {

// Data holder of Array.
//
// C++ Array and Python bindings both share ArrayBody through shared_ptr. C++ Array provides the value-based semantics of Array in C++,
// while Python Array provides the reference-based semantics, which is more natural in Python.
//
// The current design requires a subtle overhead on converting between C++ Array and Python Array (due to reference counting), which is
// currently considered to be ignorable compared to other Python operations.
//
// NOTE: This class should not be instantiated by any functions except those defined in array.cc. This class is still defined here so that
// the code is made simple and we can use inline access to each member from member accessor functions of Array.
class ArrayBody {
public:
    ArrayBody(const Shape& shape, Dtype dtype, const DeviceId& device_id, bool is_contiguous, std::shared_ptr<void> data, int64_t offset,
              std::vector<std::shared_ptr<ArrayNode>> nodes = std::vector<std::shared_ptr<ArrayNode>>());

private:
    friend class ::xchainer::Array;

    Shape shape_;
    Dtype dtype_;
    DeviceId device_id_;
    bool is_contiguous_;
    std::shared_ptr<void> data_;
    int64_t offset_;
    std::vector<std::shared_ptr<ArrayNode>> nodes_;
};

void SetUpOpNodes(const std::string& name, const std::vector<std::reference_wrapper<const Array>>& inputs, Array& out,
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
    // Deep copy ctor and copy assignment
    Array(const Array& other);

    Array(Array&& other) = default;
    Array& operator=(Array&& other) = delete;

    // TODO(hvy): Copy assignment operator is deleted to avoid performance drops due to possible unwanted copies and heavy refactorings
    // later on until the behavior is better agreed upon
    Array& operator=(const Array&) = delete;

    explicit Array(gsl::not_null<std::shared_ptr<internal::ArrayBody>> body) : body_(std::move(body)) {}

    static Array FromBuffer(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, const DeviceId& device_id = GetDefaultDeviceId());

    static Array Empty(const Shape& shape, Dtype dtype, const DeviceId& device_id = GetDefaultDeviceId());
    static Array Full(const Shape& shape, Scalar scalar, Dtype dtype, const DeviceId& device_id = GetDefaultDeviceId());
    static Array Full(const Shape& shape, Scalar scalar, const DeviceId& device_id = GetDefaultDeviceId());
    static Array Zeros(const Shape& shape, Dtype dtype, const DeviceId& device_id = GetDefaultDeviceId());
    static Array Ones(const Shape& shape, Dtype dtype, const DeviceId& device_id = GetDefaultDeviceId());

    // Creates an array which has the same shape and dtype as the other array.
    // The new array is allocated in the default device_id. The device_id of the other array
    // is ignored.
    static Array EmptyLike(const Array& array, const DeviceId& device_id = GetDefaultDeviceId());
    static Array FullLike(const Array& array, Scalar scalar, const DeviceId& device_id = GetDefaultDeviceId());
    static Array ZerosLike(const Array& array, const DeviceId& device_id = GetDefaultDeviceId());
    static Array OnesLike(const Array& array, const DeviceId& device_id = GetDefaultDeviceId());

    // Creates a copy. It will be connected to all the graphs.
    Array Copy() const;

    // Creates a copy or a view. It will be disconnected from all the graphs.
    Array AsConstant(CopyKind kind = CopyKind::kView) const;

    // Creates a copy or a view. It will be disconnected from the specified graphs.
    Array AsConstant(const std::vector<GraphId>& graph_ids, CopyKind kind = CopyKind::kView) const;

    void Fill(Scalar value);

    Array& operator+=(const Array& rhs);
    Array& operator*=(const Array& rhs);
    Array operator+(const Array& rhs) const;
    Array operator*(const Array& rhs) const;

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

    std::string ToString() const;

    const std::shared_ptr<internal::ArrayBody>& body() { return body_; }
    std::shared_ptr<const internal::ArrayBody> body() const { return body_; }
    std::shared_ptr<internal::ArrayBody>&& move_body() { return std::move(body_); }

    Dtype dtype() const { return body_->dtype_; }

    const DeviceId& device_id() const { return body_->device_id_; }

    int8_t ndim() const { return shape().ndim(); }

    const Shape& shape() const { return body_->shape_; }

    int64_t element_bytes() const { return GetElementSize(dtype()); }

    const std::shared_ptr<void>& data() { return body_->data_; }

    std::shared_ptr<const void> data() const { return body_->data_; }

    bool is_contiguous() const { return body_->is_contiguous_; }

    int64_t offset() const { return body_->offset_; }

    const std::vector<std::shared_ptr<ArrayNode>>& nodes() const { return body_->nodes_; };
    std::vector<std::shared_ptr<ArrayNode>>& nodes() { return body_->nodes_; };

private:
    Array(const Shape& shape, Dtype dtype, const DeviceId& device_id, std::shared_ptr<void> data, bool is_contiguous = true, int64_t offset = 0);

    void Add(const Array& rhs, Array& out) const;
    void Mul(const Array& rhs, Array& out) const;

    std::shared_ptr<internal::ArrayBody> body_;
};

void DebugDumpComputationalGraph(std::ostream& os, const Array& array, const GraphId& graph_id, int indent = 0);

}  // namespace xchainer
