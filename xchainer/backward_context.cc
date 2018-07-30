#include "xchainer/backward_context.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/array_node.h"
#include "xchainer/backprop_mode.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/op_node.h"
#include "xchainer/routines/creation.h"

namespace xchainer {
namespace {

using internal::ArrayBody;
using internal::ArrayNode;
using internal::OpNode;

}  // namespace

namespace internal {

GradRef::GradRef(ArrayNode& array_node) : original_grad_owner_body_{array_node.weak_body().lock()} {
    if (original_grad_owner_body_ != nullptr) {
        original_grad_ptr_ = original_grad_owner_body_->GetGrad(array_node.graph_id());
    }
}

GradRef::GradRef(nonstd::nullopt_t /*nullopt*/) : temporary_grad_{std::make_unique<nonstd::optional<Array>>()} {}

nonstd::optional<Array>& GradRef::get() {
    if (original_grad_ptr_ == nullptr) {
        if (temporary_grad_ == nullptr) {
            // Original gradient is gone and this is the first accumulation.
            // Initialize the temporary gradient.
            temporary_grad_ = std::make_unique<nonstd::optional<Array>>(nonstd::nullopt);
        }

        // Target of accumulation is the temporary gradient.
        return *temporary_grad_;
    }

    // Target of accumulation is the original gradient.
    return *original_grad_ptr_;
}

}  // namespace internal

BackwardContext::BackwardContext(
        const std::shared_ptr<OpNode>& op_node,
        const internal::OpNodeBackwardEntry& backward_entry,
        gsl::span<std::shared_ptr<ArrayNode>> prev_array_nodes,
        gsl::span<internal::GradRef*> output_grads,
        std::vector<Array>& input_grads,
        const GraphId& graph_id,
        DoubleBackpropOption double_backprop_option)
    : op_node_{op_node},
      backward_entry_{backward_entry},
      prev_array_nodes_{prev_array_nodes},
      output_grads_{output_grads},
      input_grads_{input_grads},
      zero_output_grads_{prev_array_nodes_.size()},
      graph_id_{graph_id},
      double_backprop_option_{double_backprop_option} {
    assert(op_node.get() == &backward_entry.op_node());
    assert(prev_array_nodes_.size() == output_grads_.size());
    // Input grads must be initialized with null-body arrays.
    assert(std::all_of(input_grads_.begin(), input_grads_.end(), [](const Array& g) { return internal::GetArrayBody(g) == nullptr; }));
    size_t input_count = op_node->next_array_node_count();  // Total number of input arrays including those that do not require grads.
    retained_input_array_bodies_.resize(input_count);
    retained_output_array_bodies_.resize(op_node->prev_array_node_count());  // Fill with nullptr
};

bool BackwardContext::HasOutputGrad(size_t output_index) const { return gsl::at(output_grads_, output_index)->get().has_value(); }

bool BackwardContext::is_input_grad_required(size_t input_index) const { return backward_entry_.IsGradRequired(input_index); }

const Array& BackwardContext::output_grad(size_t output_index) const {
    // If the output gradient has a propagated value, return it.
    if (HasOutputGrad(output_index)) {
        return *output_grads_[output_index]->get();
    }

    // If there already is a zero-filled gradient allocated, return it.
    assert(output_index < output_count());
    nonstd::optional<Array>& zero_grad = zero_output_grads_[output_index];
    if (zero_grad.has_value()) {
        return *zero_grad;
    }

    // Allocate new zero-filled gradient and return it.
    const internal::ArrayProps& props = op_node_->GetPrevArrayProps(output_index);
    zero_grad = Zeros(props.shape, props.dtype, props.device);
    return *zero_grad;
}

Array& BackwardContext::input_grad() {
    assert(input_grads_.size() == 1);
    return input_grad(0);
}

Array& BackwardContext::input_grad(size_t index) { return gsl::at(input_grads_, index); }

Array BackwardContext::GetRetainedInput(const RetainedInputToken& token) {
    assert(token.input_index() < op_node_->next_array_node_count());
    size_t input_index = token.input_index();

    // Retrieve the kept array body for retained output.
    // Note that it's a non-const reference so that the following logic can assign to it to keep it for the repeated retrieval of the
    // retained array.
    std::shared_ptr<ArrayBody>& kept_body = gsl::at(retained_input_array_bodies_, input_index);

    if (kept_body == nullptr) {
        // Collect the pointers to array nodes of all graphs, in the input array corresponding to input_index.
        // Never null.
        std::vector<const std::shared_ptr<ArrayNode>*> next_array_nodes;
        next_array_nodes.emplace_back(&gsl::at(op_node_->next_array_nodes(), input_index));

        for (const auto& tup : op_node_->outer_graphs_prev_array_nodes()) {
            const std::vector<std::shared_ptr<ArrayNode>>& outer_prev_array_nodes = std::get<1>(tup);
            assert(!outer_prev_array_nodes.empty());

            const std::shared_ptr<ArrayNode>& prev_array_node = outer_prev_array_nodes.front();
            assert(prev_array_node != nullptr);
            assert(prev_array_node->next_op_node() != nullptr);

            const std::shared_ptr<ArrayNode>& next_array_node = gsl::at(prev_array_node->next_op_node()->next_array_nodes(), input_index);
            assert(next_array_node != nullptr);

            next_array_nodes.emplace_back(&next_array_node);
        }

        assert(!next_array_nodes.empty());
        assert(std::all_of(next_array_nodes.begin(), next_array_nodes.end(), [](const std::shared_ptr<ArrayNode>* next_array_node) {
            return next_array_node != nullptr;
        }));

        // If the input array body is alive, use it.
        // Otherwise, create a new array body and put the nodes into it.
        std::shared_ptr<ArrayBody> array_body{};
        auto it = std::find_if(next_array_nodes.begin(), next_array_nodes.end(), [](const std::shared_ptr<ArrayNode>* next_array_node) {
            return *next_array_node != nullptr;
        });
        if (it != next_array_nodes.end()) {
            array_body = (**it)->weak_body().lock();
        }
        if (array_body == nullptr) {
            array_body = internal::CreateArrayBody(token.input_array_params());

            for (const std::shared_ptr<ArrayNode>* next_array_node : next_array_nodes) {
                if (*next_array_node != nullptr) {
                    ArrayBody::AddNode(array_body, *next_array_node);
                }
            }
        }

        assert(array_body != nullptr);
        // Cut graphs of the array body
        // TODO(hvy): Avoid temporary array
        // TODO(hvy): Avoid view
        kept_body = internal::MoveArrayBody(Array{std::move(array_body)}.MakeView());
    }

    assert(kept_body != nullptr);
    return Array{kept_body};
}

Array BackwardContext::GetRetainedOutput(const RetainedOutputToken& token) {
    assert(token.output_index() < output_count());
    size_t output_index = token.output_index();

    // Retrieve the kept array body for retained output.
    // Note that it's a non-const reference so that the following logic can assign to it to keep it for the repeated retrieval of the
    // retained array.
    std::shared_ptr<ArrayBody>& kept_body = gsl::at(retained_output_array_bodies_, output_index);

    if (kept_body == nullptr) {
        // This is the first retrieval of the retained output.
        // If the original output array body is still alive. Just make a copy of array body with restricted array nodes.
        // Otherwise, a new array body is fabricated.

        // Retrieve the array body of the original output array.
        std::shared_ptr<ArrayBody> array_body{nullptr};
        const std::shared_ptr<ArrayNode>& prev_array_node = prev_array_nodes_[output_index];
        if (prev_array_node != nullptr) {
            // array node is alive
            array_body = prev_array_node->weak_body().lock();
        }

        if (array_body == nullptr) {
            // Fabricate a new array body
            array_body = GetFabricatedArrayBodyWithNodes(token);
        }

        // If the weak ptr to old previous array node was dead, replenish it with the fabricated one.
        if (prev_array_node == nullptr) {
            prev_array_nodes_[output_index] = array_body->GetArrayNode(op_node_->graph_id());
        }

        // Cut graphs of the array body
        // TODO(hvy): Avoid temporary array
        // TODO(hvy): Avoid view
        kept_body = internal::MoveArrayBody(Array{std::move(array_body)}.MakeView());
    }

    assert(kept_body != nullptr);
    return Array{kept_body};
}

std::shared_ptr<ArrayBody> BackwardContext::GetFabricatedArrayBodyWithNodes(const RetainedOutputToken& token) const {
    std::vector<std::shared_ptr<ArrayNode>> new_prev_array_nodes;

    // Loop over outer graphs to collect array nodes corresponding to the same output index
    for (const auto& tup : op_node_->outer_graphs_prev_array_nodes()) {
        const std::vector<std::shared_ptr<ArrayNode>>& prev_array_nodes = std::get<1>(tup);
        const std::shared_ptr<ArrayNode>& prev_array_node = prev_array_nodes[token.output_index()];
        assert(prev_array_node->weak_body().expired());
        new_prev_array_nodes.emplace_back(prev_array_node);
    }

    // Collect array node of this graph.
    // If the previous array node is alive, add the node to the array body.
    // Otherwise, create a new array node out of the op node.
    {
        const std::vector<std::weak_ptr<ArrayNode>>& prev_array_nodes = op_node_->prev_array_nodes();
        std::shared_ptr<ArrayNode> prev_array_node = prev_array_nodes[token.output_index()].lock();
        if (prev_array_node == nullptr) {
            // Create mocked prev array node for "this" graph, based on the current op node
            prev_array_node = internal::FabricatePrevArrayNode(op_node_, token.output_index());
        }

        new_prev_array_nodes.emplace_back(std::move(prev_array_node));
    }

    // Create a new array body with (possibly fabricated) array nodes.
    // TODO(niboshi): Avoid unnecessary copy of array body params.
    std::shared_ptr<ArrayBody> fabricated_array_body = internal::CreateArrayBody(token.output_array_params());
    for (const std::shared_ptr<ArrayNode>& prev_array_node : new_prev_array_nodes) {
        assert(prev_array_node->weak_body().expired());
        ArrayBody::AddNode(fabricated_array_body, prev_array_node);
    }

    return fabricated_array_body;
}

size_t BackwardContext::output_count() const { return zero_output_grads_.size(); }

}  // namespace xchainer
