#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {

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

    ArrayBody(const ArrayBody&) = delete;
    ArrayBody& operator=(const ArrayBody&) = delete;

    const Shape& shape() const { return shape_; }

    const Strides& strides() const { return strides_; }

    Dtype dtype() const { return dtype_; }

    Device& device() const { return device_; }

    const std::shared_ptr<void>& data() const { return data_; }

    int64_t offset() const { return offset_; }

    const std::vector<std::shared_ptr<ArrayNode>>& nodes() const { return nodes_; }

    // TODO(niboshi): Remove this function and add another to assign an array node at a specified index.
    std::vector<std::shared_ptr<ArrayNode>>& nodes() { return nodes_; }

    const std::shared_ptr<ArrayNode>& GetArrayNode(const BackpropId& backprop_id) const {
        nonstd::optional<size_t> index = GetNodeIndex(backprop_id);
        if (!index.has_value()) {
            throw XchainerError{"Array does not require gradient for backprop id: '", backprop_id, "'."};
        }
        return nodes_[*index];
    }

    bool HasArrayNode(const BackpropId& backprop_id) const { return GetNodeIndex(backprop_id).has_value(); }

    // Adds an array node to the array body.
    // The array node must have been initialized with this array body in advance.
    // Otherwise the behavior is undefined.
    // It does nothing if an array node with the same backprop ID is already registered.
    // The returned reference is only valid until the next call of AddNode on this instance.
    static const std::shared_ptr<ArrayNode>& AddNode(const std::shared_ptr<ArrayBody>& body, std::shared_ptr<ArrayNode> array_node);

    // Creates a new array node on the specified graph.
    // XchainerError is thrown if an array node is already registered on the graph.
    // The returned reference is only valid until the next call of CreateArrayNode (or AddNode) on the same ArrayBody instance.
    static const std::shared_ptr<ArrayNode>& CreateArrayNode(const std::shared_ptr<ArrayBody>& body, const BackpropId& backprop_id);

    Params GetParams() const { return {shape_, strides_, dtype_, device_, data_, offset_}; }

    // Returns a gradient array.
    // Returns nullptr if the array does not belong to the specified graph.
    const nonstd::optional<Array>* GetGrad(const BackpropId& backprop_id) const {
        return GetGradImpl<const ArrayBody*, const nonstd::optional<Array>*>(this, backprop_id);
    }

    // Returns a gradient array.
    // Returns nullptr if the array does not belong to the specified graph.
    nonstd::optional<Array>* GetGrad(const BackpropId& backprop_id) {
        return GetGradImpl<ArrayBody*, nonstd::optional<Array>*>(this, backprop_id);
    }

    // Sets a gradient array.
    // The behavior is undefined if there is no array node for the specified graph.
    void SetGrad(Array grad, const BackpropId& backprop_id);

    // Accumulates a gradient array.
    // The behavior is undefined if there is no array node for the specified graph.
    void AccumulateGrad(Array partial_grad, const BackpropId& backprop_id);

    // Clears a gradient array.
    // XchainerError is thrown if there is no array node for the specified graph.
    void ClearGrad(const BackpropId& backprop_id);

private:
    friend std::shared_ptr<ArrayBody> CreateArrayBody(
            const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

    friend std::shared_ptr<ArrayBody> CreateArrayBody(Params params);

    ArrayBody(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

    explicit ArrayBody(Params params);

    // Asserts consistency of this instance.
    //
    // This function is no-op if NDEBUG is defined.
    void AssertConsistency() const;

    template <typename ThisPtr, typename ReturnType>
    static ReturnType GetGradImpl(ThisPtr this_ptr, const BackpropId& backprop_id);

    nonstd::optional<size_t> GetNodeIndex(const BackpropId& backprop_id) const;

    Shape shape_;
    Strides strides_;
    Dtype dtype_;
    Device& device_;
    std::shared_ptr<void> data_;
    int64_t offset_;  // in bytes

    std::vector<std::shared_ptr<ArrayNode>> nodes_;
    std::vector<std::unique_ptr<nonstd::optional<Array>>> grads_;
};

std::shared_ptr<ArrayBody> CreateArrayBody(
        const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

std::shared_ptr<ArrayBody> CreateArrayBody(ArrayBody::Params params);

}  // namespace internal
}  // namespace xchainer
