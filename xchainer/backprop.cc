#include "xchainer/backprop.h"

#include <memory>
#include <queue>
#include <unordered_map>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/op_node.h"

namespace xchainer {
namespace {

auto cmp = [](std::shared_ptr<const OpNode> lhs, std::shared_ptr<const OpNode> rhs) { return lhs->rank() < rhs->rank(); };

class BackwardImpl {
    using CandidateOpNodes = std::priority_queue<std::shared_ptr<const OpNode>, std::vector<std::shared_ptr<const OpNode>>, decltype(cmp)>;
    using PreviousArrayNodeMap = std::unordered_map<std::shared_ptr<const OpNode>, std::shared_ptr<ArrayNode>>;

public:
    BackwardImpl() : candidate_op_nodes_(cmp) {};

    void run(Array& output) {
        std::shared_ptr<ArrayNode> array_node = output.mutable_node();
        if (!array_node->grad()) {
            array_node->set_grad(Array::OnesLike(output));
        }
        BuildCandidateOpNodes(array_node);
        ProcessOpNodes();
    };

private:
    void BuildCandidateOpNodes(std::shared_ptr<ArrayNode> array_node) {
        std::shared_ptr<const OpNode> op_node = array_node->next_node();
        if (op_node) {
            PushOpNode(op_node);
            InsertPreviousArrayNode(op_node, array_node);
            auto next_array_nodes = op_node->next_nodes();
            auto backward_functions = op_node->backward_functions();
            auto next_size = next_array_nodes.size();
            for (decltype(next_size) i = 0; i < next_size; ++i) {
                if (backward_functions[i]) {
                    BuildCandidateOpNodes(next_array_nodes[i]);
                }
            }
        }
    }

    std::vector<nonstd::optional<Array>> ComputeNextGradients() {
        std::shared_ptr<const OpNode> op_node = TopOpNode();
        std::shared_ptr<ArrayNode> previous_array_node = PreviousArrayNode(op_node);

        nonstd::optional<Array> gy = previous_array_node->grad();
        assert(gy);
        previous_array_node->ClearGrad();

        std::vector<nonstd::optional<Array>> gxs;
        for (auto& backward_function : op_node->backward_functions()) {
            if (backward_function) {
                gxs.emplace_back(backward_function(*gy));
            } else {
                gxs.emplace_back(nonstd::nullopt);
            }
        }
        return gxs;
    }

    void AccumulateNextGradients(std::vector<nonstd::optional<Array>> gxs) {
        std::shared_ptr<const OpNode> op_node = TopOpNode();
        gsl::span<const std::shared_ptr<ArrayNode>> next_nodes = op_node->next_nodes();
        auto next_size = next_nodes.size();
        for (decltype(next_size) i = 0; i < next_size; ++i) {
            nonstd::optional<Array> gx = std::move(gxs[i]);
            std::shared_ptr<ArrayNode> next_node = next_nodes[i];
            if (gx) {
                const nonstd::optional<Array>& grad = next_node->grad();
                if (grad) {
                    next_node->set_grad(*grad + *gx);
                } else {
                    next_node->set_grad(std::move(*gx));
                }
            }
        }
    }

    void ProcessOpNodes() {
        while (!EmptyOpNodes()) {
            std::vector<nonstd::optional<Array>> gxs = ComputeNextGradients();
            AccumulateNextGradients(gxs);
            PopOpNode();
        }
    }

    const std::shared_ptr<const OpNode>& TopOpNode() const {
        return candidate_op_nodes_.top();
    }

    bool EmptyOpNodes() const noexcept {
        return candidate_op_nodes_.empty();
    }

    void PushOpNode(std::shared_ptr<const OpNode> op_node) {
        candidate_op_nodes_.push(std::move(op_node));
    }

    void PopOpNode() {
        candidate_op_nodes_.pop();
    }

    std::shared_ptr<ArrayNode> PreviousArrayNode(std::shared_ptr<const OpNode> op_node) const {
        auto it = previous_array_node_map_.find(op_node);
        assert(it != previous_array_node_map_.end());
        return it->second;
    }

    void InsertPreviousArrayNode(std::shared_ptr<const OpNode> op_node, std::shared_ptr<ArrayNode> array_node) {
        previous_array_node_map_.emplace(std::move(op_node), std::move(array_node));
    }

    CandidateOpNodes candidate_op_nodes_;
    PreviousArrayNodeMap previous_array_node_map_;
};

}  // namespace

void Backward(Array& output) {
    BackwardImpl impl;
    impl.run(output);
}

}  // namespace xchainer
