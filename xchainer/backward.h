#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/shape.h"

namespace xchainer {

class Array;
class ArrayNode;
class BackwardContext;
class OpNode;

using ArrayRef = std::reference_wrapper<Array>;
using ConstArrayRef = std::reference_wrapper<const Array>;

namespace internal {
class ArrayBody;
}

enum class DoubleBackpropOption : bool {
    kDisable = false,
    kEnable = true,
};

using BackwardFunction = std::function<void(BackwardContext&)>;

namespace internal {

void AccumulateGrad(nonstd::optional<Array>& target_grad, Array partial_grad, const Shape& shape, Dtype dtype, Device& device);

void SetGrad(nonstd::optional<Array>& target_grad, Array grad, const Shape& shape, Dtype dtype, Device& device);

struct ArrayProps {
    Shape shape;
    Dtype dtype;
    Device& device;
};

// Reference to the gradient array corresponding to an array node, which is valid during backward computation at most.
//
// It points to the original gradient array held by the array node's owner array body if the array body is still alive.
// Otherwise, it points to a temporary gradient array which is only valid during lifetime of this class (which means until the end of
// backward computation at most, because BackwardImpl owns instances of this class).
class GradRef {
public:
    // Initialize with alive array node.
    // The array node may or may not have gradient. If not, a temporary gradient without value will be initialized.
    explicit GradRef(ArrayNode& array_node);

    // Initialize with a temporary grad without value.
    explicit GradRef(nonstd::nullopt_t);

    GradRef(const GradRef&) = delete;
    GradRef(GradRef&&) = default;
    GradRef& operator=(const GradRef&) = delete;
    GradRef& operator=(GradRef&&) = delete;

    // Returns the reference to the gradient.
    nonstd::optional<Array>& get();

private:
    // Pointer to the original gradient held by the original input array body.
    // If the array body is gone, this pointer will be nullptr.
    nonstd::optional<Array>* original_grad_ptr_{nullptr};

    // The array body which owns the original gradient, if alive.
    // This is a keeper to prevent the gradient from being released after retrieval of the pointer.
    std::shared_ptr<internal::ArrayBody> original_grad_owner_body_{nullptr};

    // Temporary gradient instantiated only when the original array body is gone.
    std::unique_ptr<nonstd::optional<Array>> temporary_grad_;
};

}  // namespace internal

// An object used by op implementations to bridge between BackwardBuilder::RetainOutput() and BackwardContext::GetRetainedOutput().
//
// See BackwardBuilder::RetainOutput() for details.
class RetainedOutputToken {
public:
    RetainedOutputToken(std::shared_ptr<internal::ArrayBody> data_array_body, size_t output_index);

    RetainedOutputToken(const RetainedOutputToken&) = default;
    RetainedOutputToken(RetainedOutputToken&&) = default;
    RetainedOutputToken& operator=(const RetainedOutputToken&) = default;
    RetainedOutputToken& operator=(RetainedOutputToken&&) = default;

private:
    friend class xchainer::BackwardContext;

    // Returns the output index.
    // It does not necessarily correspond to the output array specified in RetainOutput(), if there are more than one output arrays with the
    // same array body.
    size_t output_index() const { return output_index_; }

    const std::shared_ptr<internal::ArrayBody>& GetFabricatedArrayBodyWithNodes(const std::shared_ptr<OpNode>& op_node) const;

    // Output data array. Initially it does not have any array nodes.
    // This array is used when retrieving retained output array, in case the array body of the original output array is no longer alive.
    // Once used, the array body `data_array_body_` points to will have array nodes of the output array for all the graphs, no matter what
    // the graph backpropagation is running on.
    std::shared_ptr<internal::ArrayBody> data_array_body_;

    size_t output_index_;
};

class BackwardContext {
public:
    // Ctor
    //
    // `input_grads_storage` is where input gradients returned by backward functions will be stored.
    // Its size must be equal to the number of input arrays whose gradients are to be returned in this single backward function (1 in most
    // ordinary functions).
    BackwardContext(
            const std::shared_ptr<OpNode>& op_node,
            gsl::span<ArrayNode*> prev_array_nodes,
            gsl::span<internal::GradRef*> prev_grads,
            std::vector<Array>& input_grads_storage,
            const GraphId& graph_id,
            DoubleBackpropOption double_backprop_option);

    // Indicates whether the next order of backward is required. It reflects DoubleBackpropOption.
    bool next_required() const { return double_backprop_option_ == DoubleBackpropOption::kEnable; }

    // Returns whether the output has a propagated gradient.
    // If there is only one output, the output always has the propagated gradient, therefore you do not have to call this function in that
    // case.
    bool HasOutputGrad(size_t output_index) const;

    // Returns the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const Array& output_grad(size_t output_index) const;

    // Returns the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const Array& output_grad() const {
        assert(prev_array_nodes_.size() == 1);
        return output_grad(0);
    }

    // Returns the reference to the input gradient.
    Array& input_grad();

    // Returns the reference to the input gradient.
    Array& input_grad(size_t index);

    // Returns the retained output array.
    // The resulting array has the same value but different array body as the actual output array.
    // It has the array node of the graph for current backward computation if and only if the double backprop option is enabled, but always
    // retains array nodes for other graphs.
    Array GetRetainedOutput(const RetainedOutputToken& token);

private:
    size_t output_count() const;

    const std::shared_ptr<OpNode>& op_node_;  // never be nullptr
    gsl::span<ArrayNode*> prev_array_nodes_;
    gsl::span<internal::GradRef*> prev_grads_;
    gsl::span<const GraphId> stop_graph_ids_;

    // A reference to the storage of input gradient arrays.
    // Gradient passed in input_grad() will be put into this storage.
    // Unset gradients will have null array body.
    std::vector<Array>& input_grads_storage_;

    // Holds zero-filled arrays for outputs without actual gradients.
    // The arrays are allocated on-demand in output_grad.
    mutable std::vector<nonstd::optional<Array>> zero_output_grads_;

    // Array bodies for retained outputs.
    // Initialized by nullptrs and populated as queried by calling GetRetainedOutput().
    std::vector<std::shared_ptr<internal::ArrayBody>> retained_output_array_bodies_;

    const GraphId& graph_id_;

    DoubleBackpropOption double_backprop_option_;
};

class BackwardBuilder {
public:
    BackwardBuilder(const char* op_name, std::initializer_list<ConstArrayRef> outputs, gsl::span<const GraphId> stop_graph_ids);
    BackwardBuilder(const char* op_name, std::initializer_list<ConstArrayRef> outputs) : BackwardBuilder{op_name, outputs, {}} {}
    BackwardBuilder(const char* op_name, const Array& output, gsl::span<const GraphId> stop_graph_ids)
        : BackwardBuilder{op_name, std::initializer_list<ConstArrayRef>{output}, stop_graph_ids} {}
    BackwardBuilder(const char* op_name, const Array& output) : BackwardBuilder{op_name, {output}, {}} {}

    // Defines a backward function with respect to specified input arrays.
    // For multi-input ops, usually this function is called for each of independent subsets of input arrays.
    void Define(std::initializer_list<ConstArrayRef> inputs, const BackwardFunction& backward_func);

    // Flags an output array to be retained for use in the backward pass.
    // Op implmentations can use this function in combination with BackwardContext::GetRetainedOutput() to retrieve output arrays in the
    // backward pass.
    //
    // If an op implementation requires the output array of the forward pass in the backward pass, it should call
    // BackwardBuilder::RetainOutput() in the forward pass and keep its return value (either assign a variable or capture by
    // value in a lambda expression). In the backward pass, it should call BackwardContext::GetRetainedOutput() with this token to retrieve
    // the output array.
    //
    // Capturing the output array directly with lambda expression would cause cyclic reference and therefore would lead to memory leak.
    //
    // Reusing the token for higher-order backward functions results in undefined behavior.
    //
    // `output` must be one of the arrays specified in the constructor of BackwardBuilder as output arrays.
    // If invalid array is specified, XchainerError will be thrown.
    RetainedOutputToken RetainOutput(const Array& output);

private:
    const char* op_name_;

    // Flag indicating whether the first Define() has been called.
    bool any_defined_{false};

    // Output arrays of the op.
    std::vector<ConstArrayRef> outputs_;

    std::vector<internal::ArrayProps> output_array_props_;

    // A collection of op nodes, each of which corresponds to a graph.
    // This record is increasingly populated as new graphs are encountered in multiple Define() calls.
    std::unordered_map<GraphId, std::shared_ptr<OpNode>> op_node_map_;

    std::vector<GraphId> stop_graph_ids_;
};

void Backward(
        const Array& output,
        const GraphId& graph_id = kDefaultGraphId,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

void Backward(
        const std::vector<ConstArrayRef>& outputs,
        const GraphId& graph_id = kDefaultGraphId,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

}  // namespace xchainer
