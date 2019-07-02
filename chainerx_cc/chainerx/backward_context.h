#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <absl/types/optional.h>
#include <absl/types/span.h>

#include "chainerx/array_fwd.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_fwd.h"
#include "chainerx/constant.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/op_node.h"

namespace chainerx {
namespace internal {

class ArrayBody;
class ArrayNode;

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
    explicit GradRef(absl::nullopt_t nullopt);

    explicit GradRef(absl::optional<Array>* grad);

    ~GradRef() = default;

    GradRef(const GradRef&) = delete;
    GradRef(GradRef&&) = default;
    GradRef& operator=(const GradRef&) = delete;
    GradRef& operator=(GradRef&&) = delete;

    // Returns the reference to the gradient.
    absl::optional<Array>& get();

private:
    // Pointer to the original gradient held by the original input array body.
    // If the array body is gone, this pointer will be nullptr.
    absl::optional<Array>* original_grad_ptr_{nullptr};

    // The array body which owns the original gradient, if alive.
    // This is a keeper to prevent the gradient from being released after retrieval of the pointer.
    std::shared_ptr<ArrayBody> original_grad_owner_body_{nullptr};

    // Temporary gradient instantiated only when the original array body is gone.
    std::unique_ptr<absl::optional<Array>> temporary_grad_;
};

}  // namespace internal

// A class that holds the context information for a backward operation such as the upstream gradients.
// An instance of this class is passed to the actual backward definition and the backward definition is responsible for updating the context
// by i.e. setting the input gradients.
//
// This class is not thread safe.
class BackwardContext {
public:
    // Ctor
    //
    // `input_grads` is where input gradients returned by backward functions will be stored.
    // Its size must be equal to the number of input arrays whose gradients are to be returned in this single backward function (1 in most
    // ordinary functions).
    BackwardContext(
            const std::shared_ptr<internal::OpNode>& op_node,
            const internal::OpNodeBackwardEntry& backward_entry,
            absl::Span<std::shared_ptr<internal::ArrayNode>> output_array_nodes,
            absl::Span<internal::GradRef*> output_grads,
            std::vector<Array>& input_grads,
            DoubleBackpropOption double_backprop_option);

    size_t input_count() const { return input_grads_.size(); }

    size_t output_count() const { return output_grads_.size(); }

    // Indicates whether the next order of backward is required. It reflects DoubleBackpropOption.
    bool next_required() const { return double_backprop_option_ == DoubleBackpropOption::kEnable; }

    // Returns whether the output has a propagated gradient.
    // If there is only one output, the output always has the propagated gradient, therefore you do not have to call this function in that
    // case.
    bool HasOutputGrad(size_t output_index) const;

    // Returns whether the input gradient is expected to be computed in the backward function.
    bool is_input_grad_required(size_t input_index) const;

    // Returns the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const absl::optional<Array>& output_grad(size_t output_index) const;

    // Returns the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const absl::optional<Array>& output_grad() const {
        CHAINERX_ASSERT(output_array_nodes_.size() == 1);
        return output_grad(0);
    }

    // Returns the reference to the input gradient.
    Array& input_grad();

    // Returns the reference to the input gradient.
    Array& input_grad(size_t index);

    // TODO(hvy): Write comment.
    Array GetRetainedInput(const RetainedInputToken& token);

    // Returns the retained output array.
    // The resulting array has the same value but different array body as the actual output array.
    // It has the array node of the graph for current backward computation if and only if the double backprop option is enabled, but always
    // retains array nodes for other graphs.
    Array GetRetainedOutput(const RetainedOutputToken& token);

private:
    std::shared_ptr<internal::ArrayBody> GetFabricatedArrayBodyWithNodes(const RetainedOutputToken& token) const;

    const std::shared_ptr<internal::OpNode>& op_node_;  // never be nullptr

    const internal::OpNodeBackwardEntry& backward_entry_;

    // Output array nodes of the op node.
    // Null if the array node is gone (the weak pointer is dead).
    absl::Span<std::shared_ptr<internal::ArrayNode>> output_array_nodes_;

    absl::Span<internal::GradRef*> output_grads_;

    // A reference to the storage of input gradient arrays.
    // Gradient passed in input_grad() will be put into this storage.
    // Unset gradients will have null array body.
    std::vector<Array>& input_grads_;

    std::vector<std::shared_ptr<internal::ArrayBody>> retained_input_array_bodies_;

    // Array bodies for retained outputs.
    // Initialized by nullptrs and populated as queried by calling GetRetainedOutput().
    std::vector<std::shared_ptr<internal::ArrayBody>> retained_output_array_bodies_;

    DoubleBackpropOption double_backprop_option_;

    // Be introduced to avoid the return value of output_grad() being destroyed at the end of the function.
    const absl::optional<Array> zero_grad_ = absl::nullopt;
};

}  // namespace chainerx
