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
namespace backward_builder_detail {

template <typename Tag>
class RetainedArrayToken {
public:
    RetainedArrayToken(internal::ArrayBody::Params array_params, size_t index) : array_params_{std::move(array_params)}, index_{index} {}

    RetainedArrayToken(const RetainedArrayToken&) = default;
    RetainedArrayToken(RetainedArrayToken&&) = default;
    RetainedArrayToken& operator=(const RetainedArrayToken&) = default;
    RetainedArrayToken& operator=(RetainedArrayToken&&) = default;

private:
    friend class xchainer::BackwardContext;

    // Returns the array index.
    size_t index() const { return index_; }

    const internal::ArrayBody::Params& array_params() const { return array_params_; }

    internal::ArrayBody::Params array_params_;

    size_t index_;
};

}  // namespace backward_builder_detail

// An object used by op implementations to bridge between BackwardBuilder::RetainInput() and BackwardContext::GetRetainedInput().
//
// See BackwardBuilder::RetainInput() for details.
using RetainedInputToken = backward_builder_detail::RetainedArrayToken<struct InputTag>;

// An object used by op implementations to bridge between BackwardBuilder::RetainOutput() and BackwardContext::GetRetainedOutput().
//
// See BackwardBuilder::RetainOutput() for details.
using RetainedOutputToken = backward_builder_detail::RetainedArrayToken<struct OutputTag>;

class BackwardBuilder {
private:
    using PrevArrayNodes = std::vector<std::shared_ptr<internal::ArrayNode>>;

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

    // TODO(hvy): Write comment.
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
        explicit operator bool() const { return is_definition_required(); }

        // Defines a backward function with respect to specified input arrays (target).
        void Define(const BackwardFunction& backward_func);

        bool is_definition_required() const { return !graph_to_next_array_nodes_.empty(); }

    private:
        friend class BackwardBuilder;  // Only BackwardBuilder can create Target

        using NextArrayNodes = std::vector<const std::shared_ptr<internal::ArrayNode>*>;

        Target(BackwardBuilder& builder, std::vector<size_t> input_indices);

        const char* op_name() { return builder_.op_name_; }
        std::vector<ConstArrayRef>& outputs() { return builder_.outputs_; }
        std::unordered_map<GraphId, std::shared_ptr<internal::OpNode>>& op_node_map() { return builder_.op_node_map_; }

        void PrepareGraphToNextArrayNodes();
        std::shared_ptr<internal::OpNode>& FindOrCreateOpNode(const GraphId& graph_id);
        void RegisterOuterGraphsPreviousArrayNodes(const std::shared_ptr<internal::OpNode>& op_nodes);

        BackwardBuilder& builder_;
<<<<<<< HEAD
        std::vector<size_t> input_indices_;
=======
        std::vector<ConstArrayRef> inputs_;
>>>>>>> 80ee7b9... Revert "wip: Let BackwardBuilder hold intermediate state of backward definition"
        std::unordered_map<GraphId, NextArrayNodes> graph_to_next_array_nodes_;
    };

    Target CreateTarget(std::vector<size_t> input_indices) { return Target{*this, std::move(input_indices)}; }
    Target CreateTarget(size_t input_index) { return Target{*this, {input_index}}; }

private:
    std::unordered_map<GraphId, std::vector<std::shared_ptr<internal::ArrayNode>>> prev_array_node_all_graphs();

    const char* op_name_;

    // Input arrays of the op.
    std::vector<ConstArrayRef> inputs_;

    // Flags indicating whether CreateTarget has been called for each of the input arrays.
    // All of these flags must be true after all the backwards have been defined for a BackwardBuilder.
    // This can be checked by calling is_complete();
    std::vector<bool> inputs_target_created_;

    // Output arrays of the op.
    std::vector<ConstArrayRef> outputs_;

    // A collection of op nodes, each of which corresponds to a graph.
    // This record is increasingly populated as new graphs are encountered in multiple Define() calls.
    std::unordered_map<GraphId, std::shared_ptr<internal::OpNode>> op_node_map_;

    std::unordered_map<GraphId, PrevArrayNodes> prev_array_node_all_graphs_;
};

}  // namespace xchainer
