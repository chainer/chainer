#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <gsl/gsl>

#include "chainerx/array.h"
#include "chainerx/array_body.h"
#include "chainerx/array_node.h"
#include "chainerx/constant.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/op_node.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace backward_builder_detail {

// This class is used by the BackwardBuilder to record retained inputs and outputs.
// The records are used to create outer graph edges (between op nodes and previous array nodes) when the builder is finalized.
class RetentionRecord {
public:
    explicit RetentionRecord(size_t size) : size_{size} { CHAINERX_ASSERT(size_ > 0); }

    size_t size() const { return size_; }

    void Record(size_t index) {
        if (flags_.empty()) {
            flags_.resize(size_);
        }
        gsl::at(flags_, index) = static_cast<int8_t>(true);
    }

    bool IsAnyRecorded() const { return !flags_.empty(); }

    bool IsRecorded(size_t index) const { return static_cast<bool>(flags_[index]); }

private:
    size_t size_{};
    std::vector<int8_t> flags_{};  // binary flags
};

template <typename Tag>
class RetainedArrayToken {
public:
    RetainedArrayToken(internal::ArrayBody::Params array_params, size_t index) : array_params_{std::move(array_params)}, index_{index} {}

    ~RetainedArrayToken() = default;

    RetainedArrayToken(const RetainedArrayToken&) = default;
    RetainedArrayToken(RetainedArrayToken&&) noexcept = default;
    RetainedArrayToken& operator=(const RetainedArrayToken&) = default;
    // TODO(hvy): Make the move assignment operator noexcept.
    RetainedArrayToken& operator=(RetainedArrayToken&&) = default;  // NOLINT(performance-noexcept-move-constructor)

private:
    friend class chainerx::BackwardContext;

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

// A class that is used to define backward operations and connect the graph.
//
// This class is not thread safe.
class BackwardBuilder {
public:
    // Target is responsible to define edges from OpNode to input ArrayNodes with given BackwardFunction.
    // Note that Targets built from the same BackwardBuilder share some properties not to compute again.
    class Target {
    public:
        explicit operator bool() const { return is_definition_required(); }

        // Defines a backward function with respect to specified input arrays (target).
        void Define(const BackwardFunction& backward_func);

        bool is_definition_required() const { return !graph_to_input_array_nodes_.empty(); }

    private:
        friend class BackwardBuilder;  // Only BackwardBuilder can create Target

        using InputArrayNodes = std::vector<const std::shared_ptr<internal::ArrayNode>*>;

        Target(BackwardBuilder& builder, std::vector<size_t> input_indices);

        // Collect input ArrayNodes, grouped by graph considering IsBackpropRequired.
        // This functions is only called once in the constructor.
        absl::flat_hash_map<BackpropId, InputArrayNodes> CreateInputArrayNodesMap() const;

        BackwardBuilder& builder_;
        std::vector<size_t> input_indices_;

        // TODO(hvy): Consider using linear search since elements are usually few.
        absl::flat_hash_map<BackpropId, InputArrayNodes> graph_to_input_array_nodes_;
    };

    // TODO(niboshi): Add an overload to accept `const std::vector<Array>&` as `inputs` and `outputs`
    // Note that simply overloading with the above type will results in ambiguous calls.
    // One solution is to define a type that accepts all of the expected types of inputs.
    BackwardBuilder(const char* op_name, std::vector<ConstArrayRef> inputs, std::vector<ConstArrayRef> outputs);
    BackwardBuilder(const char* op_name, const Array& input, std::vector<ConstArrayRef> outputs)
        : BackwardBuilder{op_name, std::vector<ConstArrayRef>{input}, std::move(outputs)} {}
    BackwardBuilder(const char* op_name, std::vector<ConstArrayRef> inputs, const Array& output)
        : BackwardBuilder{op_name, std::move(inputs), std::vector<ConstArrayRef>{output}} {}
    BackwardBuilder(const char* op_name, const Array& input, const Array& output)
        : BackwardBuilder{op_name, std::vector<ConstArrayRef>{input}, std::vector<ConstArrayRef>{output}} {}
    ~BackwardBuilder() { CHAINERX_ASSERT(is_finalized_); }

    BackwardBuilder(const BackwardBuilder&) = delete;
    BackwardBuilder(BackwardBuilder&&) noexcept = default;
    BackwardBuilder& operator=(const BackwardBuilder&) = delete;
    BackwardBuilder& operator=(BackwardBuilder&&) = delete;

    // Creates a backward target for the specified inputs.
    Target CreateTarget(std::vector<size_t> input_indices) {
        // input_indices shouldn't have duplicates.
        CHAINERX_ASSERT((std::set<size_t>{input_indices.begin(), input_indices.end()}.size() == input_indices.size()));

        for (size_t input_index : input_indices) {
            CHAINERX_ASSERT(input_index < inputs_target_created_.size());
            CHAINERX_ASSERT(!inputs_target_created_[input_index]);
            inputs_target_created_[input_index] = true;
        }
        return Target{*this, std::move(input_indices)};
    }

    // Creates a backward target for the specified input.
    Target CreateTarget(size_t input_index) { return CreateTarget(std::vector<size_t>{input_index}); }

    // Creates a backward target for all the inputs.
    Target CreateTarget() {
        std::vector<size_t> input_indices;
        input_indices.resize(inputs_.size());
        std::iota(input_indices.begin(), input_indices.end(), size_t{0});

        return CreateTarget(std::move(input_indices));
    }

    // TODO(hvy): Write comment.
    RetainedInputToken RetainInput(size_t input_index);

    std::vector<RetainedInputToken> RetainInput(std::vector<size_t> indices);

    // Flags an output array to be retained for use in the backward pass.
    // Op implementations can use this function in combination with BackwardContext::GetRetainedOutput() to retrieve output arrays in the
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
    // If invalid array is specified, ChainerxError will be thrown.
    RetainedOutputToken RetainOutput(size_t output_index);
    std::vector<RetainedOutputToken> RetainOutput(std::vector<size_t> indices);

    // Finalizes the builder.
    //
    // This functions must be called when targets have been created for all inputs.
    void Finalize();

private:
    // Create an op node for a specific graph.
    // Edges from output nodes to the op node are connected.
    std::shared_ptr<internal::OpNode>& FindOrCreateOpNode(const BackpropId& backprop_id);

    // Add shared ptrs between op nodes and array nodes belonging to outer graphs.
    // This functions is called once when the builder is finalized.
    // These references are required to restore retained inputs/outputs.
    void AddEdgesFromOpNodeToArrayNodeOfOuterGraphsForRetention();

    void ConnectBackpropIds();

    const char* op_name_;

    Context& context_;

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
    absl::flat_hash_map<BackpropId, std::shared_ptr<internal::OpNode>> op_node_map_;

    backward_builder_detail::RetentionRecord input_retention_record_;
    backward_builder_detail::RetentionRecord output_retention_record_;

    bool has_any_applicable_outputs_;
    bool is_finalized_{false};
};

}  // namespace chainerx
