#include "xchainer/backprop.h"

#include <memory>
#include <queue>
#include <set>
#include <unordered_map>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/op_node.h"

namespace xchainer {
namespace {

using PreviousNodeMap = std::unordered_map<const OpNode*, std::shared_ptr<ArrayNode>>;

}  // namespace

void Backward(Array& output) {
    std::shared_ptr<ArrayNode> node = output.mutable_node();
    auto cmp = [](std::shared_ptr<const OpNode> lhs, std::shared_ptr<const OpNode> rhs) { return lhs->rank() < rhs->rank(); };
    std::priority_queue<std::shared_ptr<const OpNode>, std::vector<std::shared_ptr<const OpNode>>, decltype(cmp)> candidate_nodes(cmp);
    std::set<const OpNode*> seen_set;
    PreviousNodeMap previous_node_map;

    // Initialize the output gradient if needed.
    if (!node->grad()) {
        node->set_grad(Array::OnesLike(output));
    }

    // Push the OpNode next to the output's ArrayNode to the priority queue.
    std::shared_ptr<const OpNode> op_node = node->next_node();
    if (op_node) {
        const OpNode* op_node_ptr = op_node.get();
        candidate_nodes.push(op_node);
        seen_set.insert(op_node_ptr);
        previous_node_map.emplace(op_node_ptr, node);
    }

    // Iteratively backprop.
    while (!candidate_nodes.empty()) {
        std::shared_ptr<const OpNode> op_node = candidate_nodes.top();
        candidate_nodes.pop();

        // TODO(takagi): Properly handle the case that the same array is passed multiple times as inputs in the same function (e.g. an
        // expression like f(x, x))

        // Get the OpNode's previous node.
        auto it = previous_node_map.find(op_node.get());
        assert(it != previous_node_map.end());
        std::shared_ptr<ArrayNode> previous_node = it->second;

        // Get the previous node's gradient.
        const nonstd::optional<Array>& gy = previous_node->grad();
        assert(gy);

        // Compute the OpNode's next gradients.
        std::vector<nonstd::optional<Array>> gxs;
        for (auto& backward_function : op_node->backward_functions()) {
            if (backward_function) {
                Array gx = backward_function(*gy);
                gxs.emplace_back(std::move(gx));
            } else {
                gxs.emplace_back(nonstd::nullopt);
            }
        }

        // Release the intermediate node's gradient.
        previous_node->ClearGrad();

        gsl::span<const std::shared_ptr<ArrayNode>> next_nodes = op_node->next_nodes();
        auto next_size = next_nodes.size();
        for (decltype(next_size) i = 0; i < next_size; ++i) {
            nonstd::optional<Array> gx = std::move(gxs[i]);
            std::shared_ptr<ArrayNode> next_node = next_nodes[i];
            if (gx) {
                // Accumulate the Opnode's next gradient.
                const nonstd::optional<Array>& grad = next_node->grad();
                if (grad) {
                    next_node->set_grad(*grad + *gx);
                } else {
                    next_node->set_grad(std::move(*gx));
                }

                // Push the next OpNode to the priority queue.
                std::shared_ptr<const OpNode> next_op_node = next_node->next_node();
                if (next_op_node) {
                    const OpNode* next_op_node_ptr = next_op_node.get();
                    if (seen_set.find(next_op_node_ptr) == seen_set.end()) {
                        candidate_nodes.push(next_op_node);
                        seen_set.insert(next_op_node_ptr);
                        previous_node_map.emplace(next_op_node_ptr, next_node);
                    }
                }
            }
        }
    }
}

}  // namespace xchainer
