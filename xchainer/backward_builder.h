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

    const internal::ArrayBody::Params& output_array_params() const { return output_array_params_; }

    internal::ArrayBody::Params output_array_params_;

    size_t output_index_;
};

class BackwardBuilder {
public:
    BackwardBuilder(const char* op_name, std::initializer_list<ConstArrayRef> outputs);
    BackwardBuilder(const char* op_name, const Array& output) : BackwardBuilder{op_name, std::initializer_list<ConstArrayRef>{output}} {}

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

        Target(BackwardBuilder& builder, std::initializer_list<ConstArrayRef> inputs);

        const char* op_name() { return builder_.op_name_; }
        bool any_defined() { return builder_.any_defined_; }
        void set_any_defined(bool defined) { builder_.any_defined_ = defined; }
        std::vector<ConstArrayRef>& outputs() { return builder_.outputs_; }
        std::unordered_map<GraphId, std::shared_ptr<internal::OpNode>>& op_node_map() { return builder_.op_node_map_; }

        void PrepareGraphToNextArrayNodes();
        std::shared_ptr<internal::OpNode>& FindOrCreateOpNode(const GraphId& graph_id);
        void RegisterOuterGraphsPreviousArrayNodes(const std::vector<internal::OpNode*>& op_nodes);

        BackwardBuilder& builder_;
        std::vector<ConstArrayRef> inputs_;
        std::unordered_map<GraphId, NextArrayNodes> graph_to_next_array_nodes_;
    };

    Target CreateTarget(std::initializer_list<ConstArrayRef> inputs) { return Target{*this, inputs}; }
    Target CreateTarget(const Array& input) { return Target{*this, {input}}; }

private:
    const char* op_name_;

    // Flag indicating whether the first Define() has been called.
    bool any_defined_{false};

    // Output arrays of the op.
    std::vector<ConstArrayRef> outputs_;

    // A collection of op nodes, each of which corresponds to a graph.
    // This record is increasingly populated as new graphs are encountered in multiple Define() calls.
    std::unordered_map<GraphId, std::shared_ptr<internal::OpNode>> op_node_map_;
};

}  // namespace xchainer
