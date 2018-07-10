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
class ArrayNode;

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
    ArrayBody(Shape shape, Strides strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

    // Adds an array node to the array body.
    // The array node must have been initialized with this array body in advance.
    // Otherwise the behavior is undefined.
    // It does nothing if an array node with the same graph ID is already registered.
    // The returned reference is only valid until the next call of AddNode on this instance.
    const std::shared_ptr<ArrayNode>& AddNode(std::shared_ptr<ArrayNode> array_node);

    // Returns a gradient array.
    // Returns nullptr if the array does not belong to the specified graph.
    const nonstd::optional<Array>* GetGrad(const GraphId& graph_id) const {
        return GetGradImpl<const ArrayBody*, const nonstd::optional<Array>*>(this, graph_id);
    }

    // Returns a gradient array.
    // Returns nullptr if the array does not belong to the specified graph.
    nonstd::optional<Array>* GetGrad(const GraphId& graph_id) { return GetGradImpl<ArrayBody*, nonstd::optional<Array>*>(this, graph_id); }

    // Sets a gradient array.
    // The behavior is undefined if there is no array node for the specified graph.
    void SetGrad(Array grad, const GraphId& graph_id);

    // Accumulates a gradient array.
    // The behavior is undefined if there is no array node for the specified graph.
    void AccumulateGrad(Array partial_grad, const GraphId& graph_id);

    // Clears a gradient array.
    // XchainerError is thrown if there is no array node for the specified graph.
    void ClearGrad(const GraphId& graph_id);

    ArrayBody(const ArrayBody&) = delete;
    ArrayBody& operator=(const ArrayBody&) = delete;

private:
    friend class ::xchainer::Array;

    // Asserts consistency of this instance.
    //
    // This function is no-op if NDEBUG is defined.
    void AssertConsistency() const;

    template <typename ThisPtr, typename ReturnType>
    static ReturnType GetGradImpl(ThisPtr this_ptr, const GraphId& graph_id);

    nonstd::optional<size_t> GetNodeIndex(const GraphId& graph_id) const;

    Shape shape_;
    Strides strides_;
    Dtype dtype_;
    Device& device_;
    std::shared_ptr<void> data_;
    int64_t offset_;  // in bytes

    std::vector<std::shared_ptr<ArrayNode>> nodes_;
    std::vector<std::unique_ptr<nonstd::optional<Array>>> grads_;
};

}  // namespace internal
}  // namespace xchainer
