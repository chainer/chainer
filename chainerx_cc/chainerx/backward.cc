#include "chainerx/backward.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/array_body.h"
#include "chainerx/array_node.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_context.h"
#include "chainerx/backward_fwd.h"
#include "chainerx/context.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/op_node.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace {

using internal::ArrayBody;
using internal::ArrayNode;
using internal::OpNode;

}  // namespace

namespace internal {
namespace {

// Throws GradientError in case of mismatch in gradient array props.
void CheckGradCompatible(const Array& grad, const Shape& shape, Dtype dtype, Device& device) {
    if (dtype != grad.dtype()) {
        throw GradientError{"Gradient dtypes do not match. Expected: ", dtype, " Actual: ", grad.dtype(), "."};
    }
    if (shape != grad.shape()) {
        throw GradientError{"Gradient shapes do not match. Expected: ", shape, " Actual: ", grad.shape(), "."};
    }
    if (&device != &grad.device()) {
        throw GradientError{"Gradient devices do not match. Expected: ", device.name(), " Actual: ", grad.device().name(), "."};
    }
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

}  // namespace internal

namespace {

struct OpNodeComparator {
    bool operator()(const std::shared_ptr<OpNode>& lhs, const std::shared_ptr<OpNode>& rhs) const { return lhs->rank() < rhs->rank(); }
};

// Pushes an ArrayNode or an OpNode to the given container if not already seen. Does nothing otherwise.
// This function is called when traversing the graph to extract a subgraph when inputs are given to backward.
// Returns true if the node was inserted, false otherwise.
template <typename T>
bool PushNodeIfNotSeen(std::vector<T*>& nodes, T* node, std::unordered_set<T*>& seen_nodes) {
    static_assert(
            std::is_same<T, ArrayNode>::value || std::is_same<T, OpNode>::value, "Only ArrayNodes or OpNodes are allowed to be pushed.");
    CHAINERX_ASSERT(node != nullptr);

    bool emplaced = seen_nodes.emplace(node).second;
    if (emplaced) {
        nodes.emplace_back(node);
    }
    return emplaced;
}

// Returns a subgraph that contains only the nodes that are necessary for computing the gradients w.r.t. the given inputs.
// The subgraph is represented by a mapping from op nodes to boolean flags corresponding their inputs.
// The flags are true for inputs that require backprop and false otherwise.
std::unordered_map<OpNode*, std::vector<uint8_t>> CreateSubgraph(
        const std::vector<ConstArrayRef>& inputs,
        const std::vector<std::reference_wrapper<const std::shared_ptr<ArrayNode>>>& output_array_nodes,
        const BackpropId& backprop_id) {
    // To extract a subgraph, the graph must be traversed "forwards" starting from the inputs towards the outputs.

    // A "forward" mapping of array nodes to op nodes of which the array nodes are inputs.
    std::unordered_multimap<ArrayNode*, OpNode*> forward_op_nodes;
    {
        std::unordered_set<OpNode*> seen_op_nodes;
        std::vector<OpNode*> candidate_op_nodes;
        candidate_op_nodes.reserve(output_array_nodes.size());

        // Initialize op node queue starting from outputs.
        for (const std::shared_ptr<ArrayNode>& array_node : output_array_nodes) {
            if (array_node != nullptr) {
                forward_op_nodes.emplace(array_node.get(), nullptr);  // Outputs have no forward op nodes.
                const std::shared_ptr<OpNode>& op_node = array_node->creator_op_node();
                if (op_node != nullptr) {
                    PushNodeIfNotSeen(candidate_op_nodes, op_node.get(), seen_op_nodes);
                }
            }
        }

        // Traverse op node queue towards inputs.
        while (!candidate_op_nodes.empty()) {
            OpNode* op_node = candidate_op_nodes.back();
            candidate_op_nodes.pop_back();

            for (const std::shared_ptr<ArrayNode>& array_node : op_node->input_array_nodes()) {
                if (array_node != nullptr) {
                    forward_op_nodes.emplace(array_node.get(), op_node);
                    const std::shared_ptr<OpNode>& creator_op_node = array_node->creator_op_node();
                    if (creator_op_node != nullptr) {  // Creator op node is nullptr for inputs which should be skipped.
                        PushNodeIfNotSeen(candidate_op_nodes, creator_op_node.get(), seen_op_nodes);
                    }
                }
            }
        }
    }

    // Create the subgraph.
    std::unordered_map<OpNode*, std::vector<uint8_t>> input_required_flags;
    {
        std::unordered_set<ArrayNode*> seen_array_nodes;
        std::vector<ArrayNode*> candidate_input_array_nodes;
        std::vector<std::shared_ptr<ArrayNode>> candidate_input_array_nodes_keep_alive;
        candidate_input_array_nodes.reserve(inputs.size());

        // Initialize array node queue with inputs.
        for (const Array& input : inputs) {
            const std::shared_ptr<ArrayBody>& array_body = internal::GetArrayBody(input);
            if (const std::shared_ptr<ArrayNode>& array_node = array_body->GetArrayNode(backprop_id)) {
                PushNodeIfNotSeen(candidate_input_array_nodes, array_node.get(), seen_array_nodes);
            }
        }

        // "Forward" traverse array node queue towards outputs.
        while (!candidate_input_array_nodes.empty()) {
            ArrayNode* array_node = candidate_input_array_nodes.back();

            auto it_op_node = forward_op_nodes.find(array_node);
            if (it_op_node == forward_op_nodes.end()) {
                // Array node is not mapped. It could be an output of an op node that was not given as output.
                candidate_input_array_nodes.pop_back();
                continue;
            }

            OpNode* op_node = it_op_node->second;
            if (op_node == nullptr) {
                candidate_input_array_nodes.pop_back();
                continue;  // Array node is an output.
            }

            std::vector<uint8_t>& flags = input_required_flags[op_node];
            if (flags.empty()) {
                flags.resize(op_node->input_array_node_count());
            }

            const std::vector<std::shared_ptr<ArrayNode>>& input_array_nodes = op_node->input_array_nodes();
            for (size_t i_input = 0; i_input < op_node->input_array_node_count(); ++i_input) {
                if (input_array_nodes[i_input].get() == array_node) {
                    flags[i_input] = static_cast<int8_t>(true);
                    // Cannot break since the same inputs may appear more than once in an op node.
                }
            }

            for (const nonstd::optional<std::weak_ptr<ArrayNode>>& output_array_node : op_node->output_array_nodes()) {
                if (output_array_node.has_value()) {
                    if (std::shared_ptr<ArrayNode> out = output_array_node->lock()) {
                        if (PushNodeIfNotSeen(candidate_input_array_nodes, out.get(), seen_array_nodes)) {
                            candidate_input_array_nodes_keep_alive.emplace_back(std::move(out));
                        }
                    }
                }
            }

            forward_op_nodes.erase(it_op_node);
        }
    }

    return input_required_flags;
}

class BackwardImpl {
public:
    BackwardImpl(
            const std::vector<ConstArrayRef>& inputs,
            const std::vector<ConstArrayRef>& outputs,
            const BackpropId& backprop_id,
            DoubleBackpropOption double_backprop,
            std::unordered_map<ArrayNode*, internal::GradRef> array_node_grad_map)
        : inputs_{inputs},
          outputs_{outputs},
          backprop_id_{backprop_id},
          double_backprop_{double_backprop},
          array_node_grad_map_{std::move(array_node_grad_map)} {
        if (!outputs_.empty()) {
            // Collect output array nodes (can be nullptr).
            output_array_nodes_.reserve(outputs.size());
            for (const Array& output : outputs) {
                output_array_nodes_.emplace_back(internal::GetArrayBody(output)->GetArrayNode(backprop_id));
            }

            // Check if backward is possible for the given graph, in this context.
            // It is not possible if a graph from an outer scope has already been backpropped.
            backprop_id.context().CheckBackpropAllowed(backprop_id);

            // Graphs for which gradients will be stopped.
            // These include the current graph that is being backpropped depending on the double backprop option, as well as all graphs
            // belonging to inner scopes, i.e. graphs with higher backprop ordinal ids.
            backprop_ids_to_stop_gradient_ = backprop_id.context().GetInnerBackpropIds(backprop_id_);
            if (double_backprop_ == DoubleBackpropOption::kDisable) {
                backprop_ids_to_stop_gradient_.emplace_back(backprop_id_);
            }

            if (!inputs.empty()) {
                input_required_flags_ = CreateSubgraph(inputs, output_array_nodes_, backprop_id);
            }
        }
    }

    BackwardImpl(
            const std::vector<ConstArrayRef>& inputs,
            const std::vector<ConstArrayRef>& outputs,
            const BackpropId& backprop_id,
            DoubleBackpropOption double_backprop)
        : BackwardImpl{inputs, outputs, backprop_id, double_backprop, {}} {}

    void Run() {
        CHAINERX_ASSERT(output_array_nodes_.size() == outputs_.size());

        // Push initial output array nodes
        for (size_t i = 0; i < outputs_.size(); ++i) {
            const Array& output = outputs_[i];
            const std::shared_ptr<ArrayNode>& array_node = gsl::at(output_array_nodes_, i);

            if (array_node != nullptr) {
                // Add GradRef for output array nodes
                auto emplace_result = array_node_grad_map_.emplace(array_node.get(), internal::GradRef{*array_node});

                // Set unset output gradients to the default value of one
                if (!emplace_result.first->second.get().has_value()) {
                    emplace_result.first->second.get() = OnesLike(output, output.device());
                }

                PushCreatorOpNode(array_node);
            }
        }

        // Backpropagation
        while (!candidate_op_nodes_.empty()) {
            std::pop_heap(candidate_op_nodes_.begin(), candidate_op_nodes_.end(), OpNodeComparator{});
            std::shared_ptr<OpNode> op_node = std::move(candidate_op_nodes_.back());
            candidate_op_nodes_.pop_back();

            // Add GradRef for input array nodes
            for (const std::shared_ptr<ArrayNode>& input_array_node : op_node->input_array_nodes()) {
                if (input_array_node != nullptr) {
                    array_node_grad_map_.emplace(input_array_node.get(), internal::GradRef{*input_array_node});
                }
            }

            // Backpropagate gradients from the output array nodes into the input array nodes.
            {
                std::vector<nonstd::optional<Array>> gxs = ComputeInputGradients(op_node);
                AccumulateInputGradients(*op_node, std::move(gxs));
            }

            // Push the creator op nodes into the queue
            for (const auto& input_array_node : op_node->input_array_nodes()) {
                if (input_array_node != nullptr) {
                    PushCreatorOpNode(input_array_node);
                }
            }

            if (double_backprop_ == DoubleBackpropOption::kDisable) {
                op_node->Unchain();
            }

            // Erase the array node's temporarily held grad
            {
                auto range = output_array_node_keeper_.equal_range(op_node.get());
                for (auto it = range.first; it != range.second; ++it) {
                    size_t n_removed = array_node_grad_map_.erase(it->second.get());
                    CHAINERX_ASSERT(n_removed > 0);
                }
            }
        }

        // Register this graph as backpropped.
        backprop_id_.context().SetBackpropDone(backprop_id_);
    }

private:
    // Runs backward functions to compute gradients of input array nodes.
    std::vector<nonstd::optional<Array>> ComputeInputGradients(const std::shared_ptr<OpNode>& op_node) {
        // A single op node has multiple backward functions, each of which computes the gradients of a subset of the inputs.
        // They are responsible for non-overlapping subsets of inputs.
        // This function calls these backward functions, collects the gradients computed by them and returns the collected gradients.
        CHAINERX_ASSERT(op_node != nullptr);

        // Output array nodes. May be nullptr if the node is gone.
        std::vector<std::shared_ptr<ArrayNode>> output_array_nodes;

        // `temp_output_grads` is a set of temporary GradRefs of this op node's output array nodes.
        // This is used for output array nodes which are either dead at the moment or alive but have not been involved in the preceding
        // backpropagation.
        // This vector is just a keeper and not used in any other way. output_grads holds the pointer to it.
        // These GradRefs are only valid in the backward functions of this op node.
        // Be careful not to cause reallocation in this vector. Otherwise the pointers would be invalidated.
        std::vector<internal::GradRef> temp_output_grads;
        temp_output_grads.reserve(op_node->output_array_nodes().size());

        std::vector<internal::GradRef*> output_grads;
        for (const nonstd::optional<std::weak_ptr<ArrayNode>>& maybe_output_array_node : op_node->output_array_nodes()) {
            std::shared_ptr<ArrayNode> output_array_node = maybe_output_array_node.has_value() ? maybe_output_array_node->lock() : nullptr;

            // Get the pointer to the output gradient.
            if (output_array_node != nullptr) {
                // Output array node is alive.
                auto it = array_node_grad_map_.find(output_array_node.get());
                if (it != array_node_grad_map_.end()) {
                    // The grad mapping has the gradient for the array node.
                    // Keep a pointer to the gradient in the map.
                    output_grads.emplace_back(&it->second);
                } else {
                    // The grad mapping has no entry for the array node.
                    // Create a new entry in temporary gradients and keep a pointer to it.
                    temp_output_grads.emplace_back(*output_array_node);
                    output_grads.emplace_back(&temp_output_grads.back());
                }
            } else {
                // Output array node is dead.
                // Keep a pointer to the temporary gradient vector.
                temp_output_grads.emplace_back(nonstd::nullopt);
                output_grads.emplace_back(&temp_output_grads.back());
            }

            output_array_nodes.emplace_back(std::move(output_array_node));
        }

        // Call the backward functions and collects their gradients.
        std::vector<nonstd::optional<Array>> input_grads;
        input_grads.resize(op_node->input_array_node_count());

        const std::vector<uint8_t>& requires_grad = input_required_flags_[op_node.get()];
        for (const internal::OpNodeBackwardEntry& backward_entry : op_node->backward_entries()) {
            // Compute and set gradients at the appropriate indices.
            if (inputs_.empty() || std::any_of(
                                           backward_entry.input_array_node_indices().begin(),
                                           backward_entry.input_array_node_indices().end(),
                                           [&requires_grad](size_t i_input) { return static_cast<bool>(requires_grad[i_input]); })) {
                CallBackwardForSubsetOfInputGradients(op_node, backward_entry, output_array_nodes, input_grads, output_grads);
            }
        }

        // Make a view if the input gradient whose array body is identical to one of other output or input gradients.
        // Otherwise modifying operations such as requiring grad on one gradient would be transferred to other gradients.
        // TODO(niboshi): View is needed to make new nodes. Come up with a solution to avoid extra backward insertion.
        for (auto it = input_grads.begin(); it != input_grads.end(); ++it) {
            if (it->has_value() &&
                IsGradientIdenticalToAnyOfOtherGradients(**it, output_array_nodes, gsl::make_span(&*input_grads.begin(), &*it))) {
                **it = (*it)->MakeView();
            }
        }

        // If the output gradients corresponding to the output array nodes are not flagged as required, clear them.
        for (const std::shared_ptr<ArrayNode>& output_array_node : output_array_nodes) {
            if (output_array_node == nullptr) {
                continue;
            }
            if (std::find_if(
                        output_array_nodes_.begin(),
                        output_array_nodes_.end(),
                        [output_array_node](const std::shared_ptr<ArrayNode>& out_node) { return output_array_node == out_node; }) ==
                output_array_nodes_.end()) {
                if (output_array_node != nullptr) {
                    std::shared_ptr<ArrayBody> body = output_array_node->weak_body().lock();
                    if (body != nullptr && !body->IsGradRequired(backprop_id_)) {
                        body->ClearGrad(backprop_id_);
                    }
                }
            }
        }

        // Erase processed OpNode from the map
        output_array_node_keeper_.erase(op_node.get());

        return input_grads;
    }

    // Calls a single backward function that computes a subset of the gradients and returns the result.
    void CallBackwardForSubsetOfInputGradients(
            const std::shared_ptr<OpNode>& op_node,
            const internal::OpNodeBackwardEntry& backward_entry,
            std::vector<std::shared_ptr<ArrayNode>>& output_array_nodes,
            std::vector<nonstd::optional<Array>>& input_grads,
            std::vector<internal::GradRef*>& output_grads) {
        // `computed_input_grads` holds the storage of gradients of all the inputs of the op node.
        // The given backward entry will compute and store a subset of those gradients.
        // The backward entry may compute and store the gradients of other inputs as well, which will be ignored.
        std::vector<Array> computed_input_grads(input_grads.size());

        // Call backward.
        BackwardContext bctx{op_node, backward_entry, output_array_nodes, output_grads, computed_input_grads, double_backprop_};
        {
            NoBackpropModeScope scope{backprop_ids_to_stop_gradient_};
            backward_entry.backward_func()(bctx);
        }

        for (size_t i_input_grad : backward_entry.input_array_node_indices()) {
            // Continue if input grad is not required.
            if (!op_node->HasInputArrayNode(i_input_grad)) {
                continue;
            }
            if (!inputs_.empty() && !static_cast<bool>(input_required_flags_[op_node.get()][i_input_grad])) {
                // Traversing through subgraph but input is not a part of it.
                continue;
            }

            Array& computed_input_grad = gsl::at(computed_input_grads, i_input_grad);
            if (internal::GetArrayBody(computed_input_grad) == nullptr) {
                // Input grad is not set by backward function
                continue;
            }

            // Set grads at the appropriate index in the vector containing all the input grads of the op node.
            {
                const std::shared_ptr<ArrayNode>& input_array_node = gsl::at(op_node->input_array_nodes(), i_input_grad);
                CHAINERX_ASSERT(input_array_node != nullptr);
                nonstd::optional<Array>& input_grad = input_grads[i_input_grad];

                try {
                    internal::SetGrad(
                            input_grad,
                            computed_input_grad,
                            input_array_node->shape(),
                            input_array_node->dtype(),
                            input_array_node->device());
                } catch (const GradientError& e) {
                    // TODO(niboshi): Use std::nested_exception
                    throw GradientError{e.what(), " Op: ", op_node->name()};
                }
            }
        }
    }

    // Returns whether the specified input gradient is identical to any of the other input gradients or output gradients.
    bool IsGradientIdenticalToAnyOfOtherGradients(
            const Array& input_grad,
            const std::vector<std::shared_ptr<ArrayNode>>& output_array_nodes,
            gsl::span<nonstd::optional<Array>> other_input_grads) {
        // TODO(niboshi): Check node identity instead of body identity.
        return std::any_of(
                       output_array_nodes.begin(),
                       output_array_nodes.end(),
                       [&input_grad, this](const std::shared_ptr<ArrayNode>& output_array_node) {
                           if (output_array_node == nullptr) {
                               return false;
                           }
                           std::shared_ptr<ArrayBody> body = output_array_node->weak_body().lock();
                           if (body == nullptr) {
                               return false;
                           }
                           const nonstd::optional<Array>* output_grad = body->GetGrad(backprop_id_);
                           return output_grad != nullptr && output_grad->has_value() &&
                                  internal::GetArrayBody(input_grad) == internal::GetArrayBody(**output_grad);
                       }) ||
               std::any_of(
                       other_input_grads.begin(), other_input_grads.end(), [&input_grad](const nonstd::optional<Array>& other_input_grad) {
                           return other_input_grad.has_value() &&
                                  internal::GetArrayBody(*other_input_grad) == internal::GetArrayBody(input_grad);
                       });
    }

    void AccumulateInputGradients(const OpNode& op_node, std::vector<nonstd::optional<Array>> gxs) {
        gsl::span<const std::shared_ptr<ArrayNode>> input_array_nodes = op_node.input_array_nodes();
        CHAINERX_ASSERT(input_array_nodes.size() == gxs.size());
        for (size_t i = 0; i < input_array_nodes.size(); ++i) {
            nonstd::optional<Array>& gx = gxs[i];
            if (gx.has_value()) {
                CHAINERX_ASSERT(input_array_nodes[i] != nullptr);
                const ArrayNode& input_array_node = *input_array_nodes[i];
                // Retrieve the pointer to the input gradient.
                internal::GradRef& input_grad = array_node_grad_map_.at(input_array_nodes[i].get());
                try {
                    internal::AccumulateGrad(
                            input_grad.get(),
                            std::move(*gx),
                            input_array_node.shape(),
                            input_array_node.dtype(),
                            input_array_node.device());
                } catch (const GradientError& e) {
                    // TODO(niboshi): Use std::nested_exception
                    throw GradientError{e.what(), " Op: ", op_node.name()};
                }
            }
        }
    }

    void PushCreatorOpNode(const std::shared_ptr<ArrayNode>& array_node) {
        // When double backprop is enabled, array_node releases the pointer to the creator op node here. After this operation, array_node
        // will look like a leaf node of the graph. Note that this move does not invalidates the array_node object itself; it is guaranteed
        // by the standard that shared_ptr becomes null after move-assigned to another.
        std::shared_ptr<OpNode> creator_op_node =
                double_backprop_ == DoubleBackpropOption::kEnable ? array_node->creator_op_node() : array_node->move_creator_op_node();

        if (creator_op_node) {
            // If inputs are specified, only push back creator op nodes that are included in the subgraph.
            if (!inputs_.empty() && input_required_flags_.find(creator_op_node.get()) == input_required_flags_.end()) {
                return;
            }

            auto range = output_array_node_keeper_.equal_range(creator_op_node.get());
            if (std::none_of(range.first, range.second, [&array_node](const auto& pair) { return pair.second == array_node; })) {
                // First appearance of the combination of op node and input array node.
                bool is_first_visit = range.first == range.second;
                output_array_node_keeper_.emplace(creator_op_node.get(), array_node);  // Iterators are invalidated here.
                if (is_first_visit) {
                    // First appearance of this op node. Push it to the queue.
                    candidate_op_nodes_.push_back(std::move(creator_op_node));
                    std::push_heap(candidate_op_nodes_.begin(), candidate_op_nodes_.end(), OpNodeComparator{});
                }
            }
        }
    }

    // Op nodes to be visited. This is a max heap ordered by the rank of each op node (see OpNodeComparator).
    std::vector<std::shared_ptr<OpNode>> candidate_op_nodes_;

    // This mapping is used to keep output array nodes alive (referenced from op nodes as weak pointers).
    std::unordered_multimap<const OpNode*, std::shared_ptr<ArrayNode>> output_array_node_keeper_;

    // Arguments to Backward().
    // Be careful that references require the referred objects alive (it should be guaranteed by Backward()).
    const std::vector<ConstArrayRef>& inputs_;
    const std::vector<ConstArrayRef>& outputs_;
    std::vector<std::reference_wrapper<const std::shared_ptr<ArrayNode>>> output_array_nodes_;
    const BackpropId& backprop_id_;
    DoubleBackpropOption double_backprop_;

    // Mapping from array nodes to the corresponding gradients. Gradients may be genuine gradients held by array bodies or temporary
    // gradients which are only valid during backward computation at most.
    std::unordered_map<ArrayNode*, internal::GradRef> array_node_grad_map_;

    std::vector<BackpropId> backprop_ids_to_stop_gradient_;

    // Represents the subgraph required for backprop in case any inputs are specified.
    // Specifically, stores for an op nodes a vector of boolean flags whether an input at that index is included in the subgraph.
    std::unordered_map<OpNode*, std::vector<uint8_t>> input_required_flags_;
};

}  // namespace

void Backward(const Array& output, const nonstd::optional<BackpropId>& backprop_id, DoubleBackpropOption double_backprop) {
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(output, backprop_id);
    std::vector<ConstArrayRef> outputs{output};  // Do not inline it; we need to guarantee that the vector is alive until Run() finishes.
    BackwardImpl{{}, outputs, actual_backprop_id, double_backprop}.Run();
}

void Backward(
        const std::vector<ConstArrayRef>& outputs, const nonstd::optional<BackpropId>& backprop_id, DoubleBackpropOption double_backprop) {
    if (outputs.empty()) {
        return;
    }
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(outputs.front().get(), backprop_id);
    BackwardImpl{{}, outputs, actual_backprop_id, double_backprop}.Run();
}

std::vector<nonstd::optional<Array>> Grad(
        const std::vector<ConstArrayRef>& outputs,
        const std::vector<ConstArrayRef>& inputs,
        const nonstd::optional<BackpropId>& backprop_id,
        DoubleBackpropOption double_backprop) {
    if (inputs.empty()) {
        return {};
    }
    if (outputs.empty()) {
        return std::vector<nonstd::optional<Array>>(inputs.size(), nonstd::nullopt);
    }

    std::vector<nonstd::optional<Array>> input_grads;
    input_grads.reserve(inputs.size());

    BackpropId actual_backprop_id = internal::GetArrayBackpropId(outputs.front().get(), backprop_id);

    // Initialize the grad map with newly created gradient arrays of the inputs.
    // The existing gradients of the inputs are thus not modified.
    std::unordered_map<ArrayNode*, internal::GradRef> array_node_grad_map;
    for (const Array& input : inputs) {
        const std::shared_ptr<ArrayBody>& array_body = internal::GetArrayBody(input);
        if (const std::shared_ptr<ArrayNode>& input_array_node = array_body->GetArrayNode(actual_backprop_id)) {
            input_grads.emplace_back(nonstd::optional<Array>{});
            array_node_grad_map.emplace(input_array_node.get(), internal::GradRef{&input_grads.back()});
        } else {
            input_grads.emplace_back(nonstd::nullopt);
        }
    }

    BackwardImpl{inputs, outputs, actual_backprop_id, double_backprop, std::move(array_node_grad_map)}.Run();

    // input_grads may contain unmodified array bodies (nullptr) for arrays that are not included in the graph.
    // Those grads are returned as nullopt.
    for (nonstd::optional<Array>& grad : input_grads) {
        if (grad.has_value() && internal::GetArrayBody(*grad) == nullptr) {
            grad = nonstd::nullopt;
        }
    }

    return input_grads;
}

}  // namespace chainerx
