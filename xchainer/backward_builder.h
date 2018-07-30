#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/array_node.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/op_node.h"
#include "xchainer/shape.h"

namespace xchainer {

// An object used by op implementations to bridge between BackwardBuilder::RetainInput() and BackwardContext::GetRetainedInput().
//
// See BackwardBuilder::RetainInput() for details.
class RetainedInputToken {
public:
    RetainedInputToken(internal::ArrayBody::Params input_array_params, size_t input_index);

    RetainedInputToken(const RetainedInputToken&) = default;
    RetainedInputToken(RetainedInputToken&&) = default;
    RetainedInputToken& operator=(const RetainedInputToken&) = default;
    RetainedInputToken& operator=(RetainedInputToken&&) = default;

private:
    friend class xchainer::BackwardContext;

    // Returns the input index.
    // It does not necessarily correspond to the input array specified in RetainInput(), if there are more than one input arrays with the
    // same array body.
    size_t input_index() const { return input_index_; }

    const internal::ArrayBody::Params& input_array_params() const { return input_array_params_; }

    internal::ArrayBody::Params input_array_params_;

    size_t input_index_;
};

// An object used by op implementations to bridge between BackwardBuilder::RetainOutput() and BackwardContext::GetRetainedOutput().
//
// See BackwardBuilder::RetainOutput() for details.
class RetainedOutputToken {
public:
    RetainedOutputToken(internal::ArrayBody::Params output_array_params, size_t output_index);

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

    const internal::ArrayBody::Params& output_array_params() const { return output_array_params_; }

    internal::ArrayBody::Params output_array_params_;

    size_t output_index_;
};

class BackwardBuilder {
public:
    BackwardBuilder(const char* op_name, std::vector<ConstArrayRef> inputs, std::vector<ConstArrayRef> outputs);
    BackwardBuilder(const char* op_name, const Array& input, std::vector<ConstArrayRef> outputs)
        : BackwardBuilder{op_name, std::vector<ConstArrayRef>{input}, std::move(outputs)} {}
    BackwardBuilder(const char* op_name, std::vector<ConstArrayRef> inputs, const Array& output)
        : BackwardBuilder{op_name, std::move(inputs), std::vector<ConstArrayRef>{output}} {}
    BackwardBuilder(const char* op_name, const Array& input, const Array& output)
        : BackwardBuilder{op_name, std::vector<ConstArrayRef>{input}, std::vector<ConstArrayRef>{output}} {}

    // Returns whether the backward definitions to cover all the input arrays have finished.
    bool is_complete() const {
        return std::all_of(inputs_target_created_.begin(), inputs_target_created_.end(), [](bool done) { return done; });
    }

    RetainedInputToken RetainInput(size_t input_index);

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

    // Target is responsible to define edges from OpNode to input ArrayNodes with given BackwardFunction.
    // Note that Targets built from the same BackwardBuilder share some properties not to compute again.
    class Target {
    public:
        explicit operator bool() const { return IsGradRequired(); }

        // Defines a backward function with respect to specified input arrays (target).
        void Define(const BackwardFunction& backward_func);

        bool IsGradRequired() const { return !graph_to_next_array_nodes_.empty(); }

    private:
        friend class BackwardBuilder;  // Only BackwardBuilder can create Target

        using NextArrayNodes = std::vector<const std::shared_ptr<internal::ArrayNode>*>;

        Target(BackwardBuilder& builder, std::vector<size_t> input_indices);

        const char* op_name() { return builder_.op_name_; }
        bool any_defined() { return builder_.any_defined_; }
        void set_any_defined(bool defined) { builder_.any_defined_ = defined; }
        std::vector<ConstArrayRef>& outputs() { return builder_.outputs_; }
        std::unordered_map<GraphId, std::shared_ptr<internal::OpNode>>& op_node_map() { return builder_.op_node_map_; }

        void PrepareGraphToNextArrayNodes();
        std::shared_ptr<internal::OpNode>& FindOrCreateOpNode(const GraphId& graph_id);
        void RegisterOuterGraphsPreviousArrayNodes(const std::vector<internal::OpNode*>& op_nodes);

        BackwardBuilder& builder_;
        std::vector<size_t> input_indices_;
        std::unordered_map<GraphId, NextArrayNodes> graph_to_next_array_nodes_;
    };

    Target CreateTarget(std::vector<size_t> input_indices) { return Target{*this, std::move(input_indices)}; }
    Target CreateTarget(size_t input_index) { return Target{*this, {input_index}}; }

private:
    const char* op_name_;

    // Flag indicating whether the first Define() has been called.
    bool any_defined_{false};

    // Input arrays of the op.
    std::vector<ConstArrayRef> inputs_;

    // Flags indicating whether CreateTarget has been called for each of the input arrays.
    // All of these flags must be true after all the backward definitions have done for a BackwardBuilder.
    // This can be check by calling is_complete();
    std::vector<bool> inputs_target_created_;

    // Output arrays of the op.
    std::vector<ConstArrayRef> outputs_;

    // A collection of op nodes, each of which corresponds to a graph.
    // This record is increasingly populated as new graphs are encountered in multiple Define() calls.
    std::unordered_map<GraphId, std::shared_ptr<internal::OpNode>> op_node_map_;
};

}  // namespace xchainer
