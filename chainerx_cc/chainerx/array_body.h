#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {

class Array;

namespace internal {

class ArrayNode;

// This class is an internal data structure which holds array data/metadata (shape, dtype, ...) and backprop graph nodes and corresponding
// gradients.
class ArrayBody {
public:
    struct Params {
        Shape shape;
        Strides strides;
        Dtype dtype;
        Device& device;
        std::shared_ptr<void> data;
        int64_t offset;
    };

    ~ArrayBody() = default;

    ArrayBody(const ArrayBody&) = delete;
    ArrayBody(ArrayBody&&) = default;
    ArrayBody& operator=(const ArrayBody&) = delete;
    ArrayBody& operator=(ArrayBody&&) = delete;

    const Shape& shape() const { return shape_; }

    const Strides& strides() const { return strides_; }

    int8_t ndim() const { return shape_.ndim(); }

    Dtype dtype() const { return dtype_; }

    Device& device() const { return device_; }

    const std::shared_ptr<void>& data() const { return data_; }

    int64_t offset() const { return offset_; }

    // Returns the list of backprop IDs whose gradients are marked as required.
    // This does not take backprop mode into account.
    const std::vector<BackpropId>& grad_required_backprop_ids() const { return grad_required_backprop_ids_; }

    const std::vector<std::shared_ptr<ArrayNode>>& nodes() const { return nodes_; }

    // TODO(niboshi): Remove this function and add another to assign an array node at a specified index.
    std::vector<std::shared_ptr<ArrayNode>>& nodes() { return nodes_; }

    int64_t GetItemSize() const { return chainerx::GetItemSize(dtype()); }

    bool IsContiguous() const { return internal::IsContiguous(shape(), strides(), GetItemSize()); }

    // Returns whether the gradient of the specified backprop ID is marked as required.
    // This does not take backprop mode into account.
    bool IsGradRequired(const BackpropId& backprop_id) const {
        backprop_id.CheckValid();
        return grad_required_backprop_ids_.end() !=
               std::find(grad_required_backprop_ids_.begin(), grad_required_backprop_ids_.end(), backprop_id);
    }

    // Mark the gradient of the specified backprop ID as required.
    // This does not take backprop mode into account.
    static void RequireGrad(const std::shared_ptr<ArrayBody>& body, const BackpropId& backprop_id) {
        backprop_id.CheckValid();
        CHAINERX_ASSERT(GetKind(body->dtype_) == DtypeKind::kFloat);

        if (body->grad_required_backprop_ids_.end() ==
            std::find(body->grad_required_backprop_ids_.begin(), body->grad_required_backprop_ids_.end(), backprop_id)) {
            body->grad_required_backprop_ids_.emplace_back(backprop_id);

            if (!body->HasArrayNode(backprop_id)) {
                CreateArrayNode(body, backprop_id);
            }
        }
    }

    int64_t GetTotalSize() const { return shape().GetTotalSize(); }

    int64_t GetNBytes() const { return GetTotalSize() * GetItemSize(); }

    const std::shared_ptr<ArrayNode>& GetArrayNode(const BackpropId& backprop_id) const {
        absl::optional<size_t> index = GetNodeIndex(backprop_id);
        if (index.has_value()) {
            return nodes_[*index];
        }

        return kNullArrayNode;
    }

    bool HasArrayNode(const BackpropId& backprop_id) const { return GetNodeIndex(backprop_id).has_value(); }

    // Adds an array node to the array body.
    // The array node must have been initialized with this array body in advance.
    // Otherwise the behavior is undefined.
    // It does nothing if an array node with the same backprop ID is already registered.
    // The returned reference is only valid until the next call of AddNode on this instance.
    static const std::shared_ptr<ArrayNode>& AddNode(const std::shared_ptr<ArrayBody>& body, std::shared_ptr<ArrayNode> array_node);

    // Creates a new array node on the specified graph.
    // ChainerxError is thrown if an array node is already registered on the graph.
    // The returned reference is only valid until the next call of CreateArrayNode (or AddNode) on the same ArrayBody instance.
    static const std::shared_ptr<ArrayNode>& CreateArrayNode(const std::shared_ptr<ArrayBody>& body, const BackpropId& backprop_id);

    Params GetParams() const { return {shape_, strides_, dtype_, device_, data_, offset_}; }

    // Returns a gradient array.
    // Returns nullptr if the array does not belong to the specified graph.
    const absl::optional<Array>* GetGrad(const BackpropId& backprop_id) const {
        return GetGradImpl<const ArrayBody*, const absl::optional<Array>*>(this, backprop_id);
    }

    // Returns a gradient array.
    // Returns nullptr if the array does not belong to the specified graph.
    absl::optional<Array>* GetGrad(const BackpropId& backprop_id) {
        return GetGradImpl<ArrayBody*, absl::optional<Array>*>(this, backprop_id);
    }

    // Sets a gradient array.
    // The behavior is undefined if there is no array node for the specified graph.
    void SetGrad(Array grad, const BackpropId& backprop_id);

    // Clears a gradient array.
    // The behavior is undefined if there is no array node for the specified graph.
    void ClearGrad(const BackpropId& backprop_id);

private:
    friend std::shared_ptr<ArrayBody> CreateArrayBody(
            const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

    friend std::shared_ptr<ArrayBody> CreateArrayBody(Params params);

    ArrayBody(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

    explicit ArrayBody(Params params);

    // Asserts consistency of this instance.
    //
    // This function is no-op if CHAINERX_DEBUG is set.
    void AssertConsistency() const;

    template <typename ThisPtr, typename ReturnType>
    static ReturnType GetGradImpl(ThisPtr this_ptr, const BackpropId& backprop_id);

    absl::optional<size_t> GetNodeIndex(const BackpropId& backprop_id) const;

    // The use of non-POD static storage object here is safe, because destructing a shared_ptr with nullptr does not incur any
    // destruction order problem.
    static const std::shared_ptr<ArrayNode> kNullArrayNode;

    Shape shape_;
    Strides strides_;
    Dtype dtype_;
    Device& device_;
    std::shared_ptr<void> data_;
    int64_t offset_;  // in bytes

    std::vector<BackpropId> grad_required_backprop_ids_;
    std::vector<std::shared_ptr<ArrayNode>> nodes_;
    std::vector<std::unique_ptr<absl::optional<Array>>> grads_;
};

std::shared_ptr<ArrayBody> CreateArrayBody(
        const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

std::shared_ptr<ArrayBody> CreateArrayBody(ArrayBody::Params params);

}  // namespace internal
}  // namespace chainerx
