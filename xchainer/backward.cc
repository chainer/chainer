#include "xchainer/backward.h"

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
namespace internal {
namespace {

void CheckGradCompatible(const Array& grad, const Shape& shape, Dtype dtype, Device& device) {
    CheckEqual(dtype, grad.dtype());
    CheckEqual(shape, grad.shape());
    CheckEqual(device, grad.device());
}

}  // namespace

void AccumulateGrad(nonstd::optional<Array>& target_grad, Array partial_grad, const Shape& shape, Dtype dtype, Device& device) {
    CheckGradCompatible(partial_grad, shape, dtype, device);
    if (target_grad.has_value()) {
        target_grad = *target_grad + partial_grad;
    } else {
        target_grad = std::move(partial_grad);
    }
}

void SetGrad(nonstd::optional<Array>& target_grad, Array grad, const Shape& shape, Dtype dtype, Device& device) {
    CheckGradCompatible(grad, shape, dtype, device);
    target_grad = std::move(grad);
}

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
        gsl::span<std::shared_ptr<ArrayNode>> prev_array_nodes,
        gsl::span<internal::GradRef*> output_grads,
        std::vector<Array>& input_grads,
        const std::vector<bool>& is_input_grads_required,
        const GraphId& graph_id,
        DoubleBackpropOption double_backprop_option)
    : op_node_{op_node},
      prev_array_nodes_{prev_array_nodes},
      output_grads_{output_grads},
      input_grads_{input_grads},
      is_input_grads_required_{is_input_grads_required},
      zero_output_grads_{prev_array_nodes_.size()},
      graph_id_{graph_id},
      double_backprop_option_{double_backprop_option} {
    assert(prev_array_nodes_.size() == output_grads_.size());
    // Input grads must be initialized with null-body arrays.
    assert(std::all_of(input_grads_.begin(), input_grads_.end(), [](const Array& g) { return g.body() == nullptr; }));

    retained_output_array_bodies_.resize(op_node->prev_array_node_count());  // Fill with nullptr
};

bool BackwardContext::HasOutputGrad(size_t output_index) const { return gsl::at(output_grads_, output_index)->get().has_value(); }

bool BackwardContext::is_input_grad_required(size_t input_index) const {
    assert(input_index < is_input_grads_required_.size());
    return is_input_grads_required_[input_index];
}

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

Array BackwardContext::GetRetainedOutput(const RetainedOutputToken& token) {
    assert(token.output_index() < output_count());
    size_t output_index = token.output_index();

    // Retrieve the kept array body for retained output.
    // Note that it's a non-const reference so that the following logic can assign to it to keep it for the repeated retrieval of the
    // retained array.
    std::shared_ptr<internal::ArrayBody>& kept_body = retained_output_array_bodies_[output_index];

    if (kept_body == nullptr) {
        // This is the first retrieval of the retained output.
        // If the original output array body is still alive. Just make a copy of array body with restricted array nodes.
        // Otherwise, a new array body is fabricated.

        // Retrieve the array body of the original output array.
        std::shared_ptr<internal::ArrayBody> array_body{nullptr};
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
            // TODO(niboshi): Avoid temporary array
            prev_array_nodes_[output_index] = internal::GetMutableArrayNode(Array{array_body}, op_node_->graph_id());
        }

        // Cut graphs of the array body
        // TODO(hvy): Avoid temporary array
        // TODO(hvy): Avoid view
        kept_body = Array{std::move(array_body)}.MakeView().move_body();
    }

    assert(kept_body != nullptr);
    return Array{kept_body};
}

std::shared_ptr<internal::ArrayBody> BackwardContext::GetFabricatedArrayBodyWithNodes(const RetainedOutputToken& token) const {
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
    auto fabricated_array_body = std::make_shared<internal::ArrayBody>(token.output_array_params());
    for (const std::shared_ptr<ArrayNode>& prev_array_node : new_prev_array_nodes) {
        assert(prev_array_node->weak_body().expired());
        internal::ArrayBody::AddNode(fabricated_array_body, prev_array_node);
    }

    return fabricated_array_body;
}

size_t BackwardContext::output_count() const { return zero_output_grads_.size(); }

namespace {

struct OpNodeComparator {
    bool operator()(const std::shared_ptr<OpNode>& lhs, const std::shared_ptr<OpNode>& rhs) const { return lhs->rank() < rhs->rank(); }
};

class BackwardImpl {
public:
    BackwardImpl(const std::vector<ConstArrayRef>& outputs, const GraphId& graph_id, DoubleBackpropOption double_backprop)
        : outputs_{outputs}, graph_id_{graph_id}, double_backprop_{double_backprop} {
        for (const Array& output : outputs) {
            if (!output.IsGradRequired(graph_id)) {
                throw XchainerError{"Cannot start backprop from an array whose gradient is not required (on graph '", graph_id, "')"};
            }
            output_array_nodes_.emplace_back(internal::GetMutableArrayNode(output, graph_id));
        }

        // Check if backward is possible for the given graph, in this context.
        // It is not possible if a graph from an outer scope has already been backpropped.
        graph_id.context().CheckBackpropAllowed(graph_id);
    }

    void Run() {
        Context& context = graph_id_.context();

        // Push initial output array nodes
        for (size_t i = 0; i < outputs_.size(); ++i) {
            const Array& output = outputs_[i];
            const std::shared_ptr<ArrayNode>& array_node = output_array_nodes_[i];

            // Add GradRef for output array nodes
            auto emplace_result = array_node_grad_map_.emplace(array_node.get(), internal::GradRef{*array_node});

            // Set unset output gradients to the default value of one
            if (!emplace_result.first->second.get().has_value()) {
                emplace_result.first->second.get() = OnesLike(output, output.device());
            }

            PushNextOpNode(array_node);
        }

        // Graphs for which gradients will be stopped.
        // These include the current graph that is being backpropped depending on the double backprop option, as well as all graphs
        // belonging to inner scopes, i.e. graphs with higher graph sub ids.
        std::vector<GraphId> graph_ids_to_stop_gradient = context.GetInnerGraphIds(graph_id_);
        if (double_backprop_ == DoubleBackpropOption::kDisable) {
            graph_ids_to_stop_gradient.emplace_back(graph_id_);
        }

        // Backpropagation
        while (!candidate_op_nodes_.empty()) {
            std::pop_heap(candidate_op_nodes_.begin(), candidate_op_nodes_.end(), OpNodeComparator{});
            std::shared_ptr<OpNode> op_node = std::move(candidate_op_nodes_.back());
            candidate_op_nodes_.pop_back();

            // Add GradRef for next array nodes
            for (const std::shared_ptr<ArrayNode>& next_array_node : op_node->next_array_nodes()) {
                assert(next_array_node != nullptr);
                array_node_grad_map_.emplace(next_array_node.get(), internal::GradRef{*next_array_node});
            }

            // Backpropagate gradients from the previous array nodes into the next array nodes.
            {
                std::vector<nonstd::optional<Array>> gxs = ComputeNextGradients(op_node, graph_ids_to_stop_gradient);
                AccumulateNextGradients(*op_node, std::move(gxs));
            }

            // Push the next op nodes into the queue
            for (const auto& next_array_node : op_node->next_array_nodes()) {
                PushNextOpNode(next_array_node);
            }

            if (double_backprop_ == DoubleBackpropOption::kDisable) {
                op_node->Unchain();
            }

            // Erase the array node's temporarily held grad
            {
                auto range = previous_array_node_keeper_.equal_range(op_node.get());
                for (auto it = range.first; it != range.second; ++it) {
                    size_t n_removed = array_node_grad_map_.erase(it->second.get());
                    (void)n_removed;  // unused
                    assert(n_removed > 0);
                }
            }
        }

        // Register this graph as backpropped.
        context.SetBackpropDone(graph_id_);
    }

private:
    std::vector<nonstd::optional<Array>> ComputeNextGradients(
            const std::shared_ptr<OpNode>& op_node, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        assert(op_node != nullptr);

        // Run backward functions to compute gradients of next array nodes.
        std::vector<nonstd::optional<Array>> input_grads;
        input_grads.resize(op_node->next_array_node_count());

        // Previous array nodes. May be nullptr if the node is gone.
        std::vector<std::shared_ptr<ArrayNode>> prev_array_nodes;

        // `temp_output_grads` is a set of temporary GradRefs of this op node's previous array nodes.
        // This is used for previous array nodes which are either dead at the moment or alive but have not been involved in the preceding
        // backpropagation.
        // This vector is just a keeper and not used in any other way. output_grads holds the pointer to it.
        // These GradRefs are only valid in the backward functions of this op node.
        // Be careful not to cause reallocation in this vector. Otherwise the pointers would be invalidated.
        std::vector<internal::GradRef> temp_output_grads;
        temp_output_grads.reserve(op_node->prev_array_nodes().size());

        std::vector<internal::GradRef*> output_grads;
        for (const std::weak_ptr<ArrayNode>& maybe_prev_array_node : op_node->prev_array_nodes()) {
            std::shared_ptr<ArrayNode> prev_array_node = maybe_prev_array_node.lock();

            // Get the pointer to the previous gradient.
            if (prev_array_node != nullptr) {
                // Previous array node is alive.
                auto it = array_node_grad_map_.find(prev_array_node.get());
                if (it != array_node_grad_map_.end()) {
                    // The grad mapping has the gradient for the array node.
                    // Keep a pointer to the gradient in the map.
                    output_grads.emplace_back(&it->second);
                } else {
                    // The grad mapping has no entry for the array node.
                    // Create a new entry in temporary gradients and keep a pointer to it.
                    temp_output_grads.emplace_back(*prev_array_node);
                    output_grads.emplace_back(&temp_output_grads.back());
                }
            } else {
                // Previous array node is dead.
                // Keep a pointer to the temporary gradient vector.
                temp_output_grads.emplace_back(nonstd::nullopt);
                output_grads.emplace_back(&temp_output_grads.back());
            }

            prev_array_nodes.emplace_back(std::move(prev_array_node));
        }

        for (const internal::OpNodeBackwardEntry& backward_entry : op_node->backward_entries()) {
            const size_t input_count = backward_entry.next_array_node_count();

            // `input_grads_subset` stores the next gradients (`input_grads`) of the subset of input arrays of this backward
            // call. `BackwardContext` holds it by reference and assignment to BackwardContext::input_grad() stores the
            // gradients there. It initially holds null-body arrays.
            std::vector<Array> input_grads_subset;
            input_grads_subset.resize(input_count);

            // Boolean flags indicating whether the next grads are required to be calculated in the backward function.
            std::vector<bool> is_input_grads_required;
            is_input_grads_required.reserve(input_count);
            std::transform(
                    backward_entry.next_array_node_indices().begin(),
                    backward_entry.next_array_node_indices().end(),
                    std::back_inserter(is_input_grads_required),
                    [](nonstd::optional<size_t> i_input_grad) { return i_input_grad.has_value(); });

            // Call backward.
            BackwardContext bctx{
                    op_node, prev_array_nodes, output_grads, input_grads_subset, is_input_grads_required, graph_id_, double_backprop_};
            {
                NoBackpropModeScope scope{graph_ids_to_stop_gradient};
                backward_entry.backward_func()(bctx);
            }

            for (size_t i_input = 0; i_input < input_count; ++i_input) {
                if (!is_input_grads_required[i_input]) {
                    // Input grad is not required
                    continue;
                }

                Array& input_grad = gsl::at(input_grads_subset, i_input);
                if (input_grad.body() == nullptr) {
                    // Input grad is not set by backward function
                    continue;
                }

                // Make a view if the next gradient is identical to one of other prev or next gradients.
                // TODO(niboshi): Check node identity instead of body identity.
                if (std::any_of(
                            prev_array_nodes.begin(),
                            prev_array_nodes.end(),
                            [&input_grad, this](const std::shared_ptr<ArrayNode>& prev_array_node) {
                                if (prev_array_node == nullptr) {
                                    return false;
                                }
                                std::shared_ptr<internal::ArrayBody> body = prev_array_node->weak_body().lock();
                                if (body == nullptr) {
                                    return false;
                                }
                                const nonstd::optional<Array>* prev_grad = body->GetGrad(graph_id_);
                                return prev_grad != nullptr && prev_grad->has_value() && input_grad.body() == (*prev_grad)->body();
                            }) ||
                    std::any_of(
                            input_grads_subset.begin(),
                            input_grads_subset.begin() + i_input,
                            [&input_grad](const Array& another_input_grad) { return another_input_grad.body() == input_grad.body(); })) {
                    // TODO(niboshi): View is needed to make new nodes. Come up with a solution to avoid extra backward insertion.
                    input_grad = input_grad.MakeView();
                }

                // Accumulate grads.
                {
                    nonstd::optional<size_t> i_input_grad = backward_entry.next_array_node_indices()[i_input];
                    assert(i_input_grad.has_value());

                    nonstd::optional<Array>& target_grad = input_grads[*i_input_grad];
                    const ArrayNode& next_array_node = *op_node->next_array_nodes()[*i_input_grad];

                    internal::AccumulateGrad(
                            target_grad, input_grad, next_array_node.shape(), next_array_node.dtype(), next_array_node.device());
                }
            }
        }

        // If previous array nodes are not output nodes of backward, clear their gradients
        for (const std::shared_ptr<ArrayNode>& prev_array_node : prev_array_nodes) {
            if (prev_array_node == nullptr) {
                continue;
            }
            if (std::find_if(
                        output_array_nodes_.begin(),
                        output_array_nodes_.end(),
                        [prev_array_node](const std::shared_ptr<ArrayNode>& out_node) { return prev_array_node == out_node; }) ==
                output_array_nodes_.end()) {
                if (prev_array_node != nullptr) {
                    std::shared_ptr<internal::ArrayBody> body = prev_array_node->weak_body().lock();
                    if (body != nullptr) {
                        body->ClearGrad(prev_array_node->graph_id());
                    }
                }
            }
        }

        // Erase processed OpNode from the map
        previous_array_node_keeper_.erase(op_node.get());

        return input_grads;
    }

    void AccumulateNextGradients(const OpNode& op_node, std::vector<nonstd::optional<Array>> gxs) {
        gsl::span<const std::shared_ptr<ArrayNode>> next_array_nodes = op_node.next_array_nodes();
        assert(next_array_nodes.size() == gxs.size());
        for (size_t i = 0; i < next_array_nodes.size(); ++i) {
            const ArrayNode& next_array_node = *next_array_nodes[i];
            nonstd::optional<Array>& gx = gxs[i];
            if (gx.has_value()) {
                // Retrieve the pointer to the next gradient.
                internal::GradRef& input_grad = array_node_grad_map_.at(next_array_nodes[i].get());
                internal::AccumulateGrad(
                        input_grad.get(), std::move(*gx), next_array_node.shape(), next_array_node.dtype(), next_array_node.device());
            }
        }
    }

    void PushNextOpNode(const std::shared_ptr<ArrayNode>& array_node) {
        // When double backprop is enabled, array_node releases the pointer to the next node here. After this operation, array_node will
        // look like a leaf node of the graph. Note that this move does not invalidates the array_node object itself; it is guaranteed
        // by the standard that shared_ptr becomes null after move-assigned to another.
        std::shared_ptr<OpNode> next_op_node =
                double_backprop_ == DoubleBackpropOption::kEnable ? array_node->next_op_node() : array_node->move_next_op_node();

        if (next_op_node) {
            auto range = previous_array_node_keeper_.equal_range(next_op_node.get());
            if (std::none_of(range.first, range.second, [&array_node](const auto& pair) { return pair.second == array_node; })) {
                // First appearance of the combination of op node and next node.
                bool is_first_visit = range.first == range.second;
                previous_array_node_keeper_.emplace(next_op_node.get(), array_node);  // Iterators are invalidated here.
                if (is_first_visit) {
                    // First appearance of this op node. Push it to the queue.
                    candidate_op_nodes_.push_back(std::move(next_op_node));
                    std::push_heap(candidate_op_nodes_.begin(), candidate_op_nodes_.end(), OpNodeComparator{});
                }
            }
        }
    }

    // Op nodes to be visited. This is a max heap ordered by the rank of each op node (see OpNodeComparator).
    std::vector<std::shared_ptr<OpNode>> candidate_op_nodes_;

    // This mapping is used to keep previous array nodes alive (referenced from op nodes as weak pointers).
    std::unordered_multimap<const OpNode*, std::shared_ptr<ArrayNode>> previous_array_node_keeper_;

    // Mapping from array nodes to the corresponding gradients. Gradients may be genuine gradients held by array bodies or temporary
    // gradients which are only valid during backward computation at most.
    std::unordered_map<ArrayNode*, internal::GradRef> array_node_grad_map_;

    // Arguments to Backward().
    // Be careful that references require the referred objects alive (it should be guaranteed by Backward()).
    const std::vector<ConstArrayRef>& outputs_;
    std::vector<std::reference_wrapper<const std::shared_ptr<ArrayNode>>> output_array_nodes_;
    const GraphId& graph_id_;
    DoubleBackpropOption double_backprop_;
};

}  // namespace

void Backward(const Array& output, const nonstd::optional<GraphId>& graph_id, DoubleBackpropOption double_backprop) {
    GraphId actual_graph_id = graph_id.has_value() ? *graph_id : output.device().context().default_graph_id();
    std::vector<ConstArrayRef> outputs{output};  // Do not inline it; we need to guarantee that the vector is alive until Run() finishes.
    BackwardImpl{outputs, actual_graph_id, double_backprop}.Run();
}

void Backward(const std::vector<ConstArrayRef>& outputs, const nonstd::optional<GraphId>& graph_id, DoubleBackpropOption double_backprop) {
    if (outputs.empty()) {
        return;
    }
    GraphId actual_graph_id = graph_id.has_value() ? *graph_id : outputs.front().get().device().context().default_graph_id();
    BackwardImpl{outputs, actual_graph_id, double_backprop}.Run();
}

}  // namespace xchainer
