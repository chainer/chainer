#include "xchainer/backprop.h"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/op_node.h"

namespace xchainer {
namespace {

struct OpNodeComparator {
    bool operator()(const std::shared_ptr<OpNode>& lhs, const std::shared_ptr<OpNode>& rhs) const { return lhs->rank() < rhs->rank(); };
};

class BackwardImpl {
public:
    BackwardImpl(const std::vector<ConstArrayRef>& outputs, const GraphId& graph_id, DoubleBackpropOption double_backprop)
        : outputs_(outputs), graph_id_(graph_id), double_backprop_(double_backprop) {
        for (const Array& output : outputs) {
            output_array_nodes_.push_back(internal::GetMutableArrayNode(output, graph_id));
        }
    };

    void Run() {
        for (size_t i = 0; i < outputs_.size(); ++i) {
            const std::shared_ptr<ArrayNode>& array_node = output_array_nodes_[i];
            if (!array_node->grad()) {
                array_node->set_grad(Array::OnesLike(outputs_[i]));
            }
            PushNextOpNode(array_node);
        }

        while (!candidate_op_nodes_.empty()) {
            std::pop_heap(candidate_op_nodes_.begin(), candidate_op_nodes_.end(), OpNodeComparator{});
            std::shared_ptr<OpNode> op_node = std::move(candidate_op_nodes_.back());
            candidate_op_nodes_.pop_back();

            std::vector<Array> gxs = ComputeNextGradients(*op_node, graph_id_);
            AccumulateNextGradients(*op_node, gxs);

            for (const auto& next_array_node : op_node->next_nodes()) {
                PushNextOpNode(next_array_node);
            }

            if (double_backprop_ == DoubleBackpropOption::kDisable) {
                op_node->Unchain();
            }
        }
    };

private:
    std::vector<Array> ComputeNextGradients(const OpNode& op_node, const GraphId& graph_id) {
        auto it = previous_array_node_map_.find(&op_node);
        assert(it != previous_array_node_map_.end());
        std::shared_ptr<ArrayNode> previous_array_node = std::move(it->second);
        previous_array_node_map_.erase(it);

        const nonstd::optional<Array>& gy = previous_array_node->grad();
        assert(gy);

        std::vector<GraphId> graph_ids_to_stop_gradient;
        if (double_backprop_ == DoubleBackpropOption::kDisable) {
            graph_ids_to_stop_gradient.emplace_back(graph_id);
        }

        std::vector<Array> gxs;
        for (const auto& backward_function : op_node.backward_functions()) {
            gxs.emplace_back(backward_function(*gy, graph_ids_to_stop_gradient));
        }

        // This may be slow if output_array_nodes_.size() (== outputs_.size()) is large.
        if (std::find_if(output_array_nodes_.begin(), output_array_nodes_.end(), [&previous_array_node](const auto& ref) {
                return previous_array_node == ref.get();
            }) == output_array_nodes_.end()) {
            previous_array_node->ClearGrad();
        }

        return gxs;
    }

    void AccumulateNextGradients(const OpNode& op_node, const std::vector<Array>& gxs) {
        gsl::span<const std::shared_ptr<ArrayNode>> next_array_nodes = op_node.next_nodes();
        assert(next_array_nodes.size() == gxs.size());
        for (size_t i = 0; i < next_array_nodes.size(); ++i) {
            const Array& gx = gxs[i];
            const std::shared_ptr<ArrayNode>& next_array_node = next_array_nodes[i];
            const nonstd::optional<Array>& grad = next_array_node->grad();
            if (grad) {
                next_array_node->set_grad(*grad + gx);
            } else {
                next_array_node->set_grad(std::move(gx));
            }
        }
    }

    void PushNextOpNode(const std::shared_ptr<ArrayNode>& array_node) {
        // When double backprop is enabled, array_node releases the pointer to the next node here. After this operation, array_node will
        // look like a leaf node of the graph. Note that this move does not invalidates the array_node object itself; it is guaranteed by
        // the standard that shared_ptr becomes null after move-assigned to another.
        std::shared_ptr<OpNode> next_op_node =
            double_backprop_ == DoubleBackpropOption::kEnable ? array_node->next_node() : array_node->move_next_node();

        if (next_op_node) {
            if (previous_array_node_map_.find(next_op_node.get()) == previous_array_node_map_.end()) {
                previous_array_node_map_.emplace(next_op_node.get(), array_node);
                candidate_op_nodes_.push_back(std::move(next_op_node));
                std::push_heap(candidate_op_nodes_.begin(), candidate_op_nodes_.end(), OpNodeComparator{});
            }
        }
    }

    // Op nodes to be visited. This is a max heap ordered by the rank of each op node (see OpNodeComparator).
    std::vector<std::shared_ptr<OpNode>> candidate_op_nodes_;

    // This mapping is used to call the backward of the op node. Using raw pointers here is safe because the op node is alive when we call
    // the backward. This mapping also works as a bookkeeping of op nodes that have already been seen.
    std::unordered_map<const OpNode*, std::shared_ptr<ArrayNode>> previous_array_node_map_;

    // Arguments to Backward()
    const std::vector<ConstArrayRef>& outputs_;
    std::vector<std::reference_wrapper<const std::shared_ptr<ArrayNode>>> output_array_nodes_;
    const GraphId& graph_id_;
    DoubleBackpropOption double_backprop_;
};

}  // namespace

void Backward(const Array& output, const GraphId& graph_id, DoubleBackpropOption double_backprop) {
    // TODO(takagi): Operations that have multiple outputs
    BackwardImpl{{output}, graph_id, double_backprop}.Run();
}

void Backward(const std::vector<Array>& outputs, const GraphId& graph_id, DoubleBackpropOption double_backprop) {
    BackwardImpl{{outputs.begin(), outputs.end()}, graph_id, double_backprop}.Run();
}

void Backward(const std::vector<ConstArrayRef>& outputs, const GraphId& graph_id, DoubleBackpropOption double_backprop) {
    BackwardImpl{outputs, graph_id, double_backprop}.Run();
}

}  // namespace xchainer
