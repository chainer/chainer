#include "xchainer/backprop_mode.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace backprop_mode_detail {

thread_local BackpropModeStack* t_backprop_mode_stack{nullptr};

template <bool kModeFlag>
void BackpropModeScope<kModeFlag>::InitializeBackpropModeStack() {
    // The outer-most scope creates an instance of BackpropModeStack.
    if (t_backprop_mode_stack == nullptr) {
        t_backprop_mode_stack = new BackpropModeStack{};
        is_outermost_ = true;
    }
}

template <bool kModeFlag>
BackpropModeScope<kModeFlag>::BackpropModeScope(Context& context) : n_{1} {
    InitializeBackpropModeStack();
    t_backprop_mode_stack->emplace_back(context, kModeFlag);
}

template <bool kModeFlag>
BackpropModeScope<kModeFlag>::BackpropModeScope(const std::vector<GraphId>& graph_ids) : n_{graph_ids.size()} {
    // Need to throw before initializing because thowing error at ctor does not call the dtor.
    for (const GraphId& graph_id : graph_ids) {
        if (&graph_ids.front().context() != &graph_id.context()) {
            throw ContextError{"Cannot specify graph ids with different contexts together."};
        }
    }
    InitializeBackpropModeStack();
    for (const GraphId& graph_id : graph_ids) {
        t_backprop_mode_stack->emplace_back(graph_id, kModeFlag);
    }
}

template <bool kModeFlag>
BackpropModeScope<kModeFlag>::~BackpropModeScope() {
    assert(t_backprop_mode_stack != nullptr);
    assert(t_backprop_mode_stack->size() >= n_);

    // Recover thread local variable to nullptr on exiting from the outer-most scope.
    if (is_outermost_) {
        assert(t_backprop_mode_stack->size() == n_);
        delete t_backprop_mode_stack;
        t_backprop_mode_stack = nullptr;
    } else {
        t_backprop_mode_stack->erase(t_backprop_mode_stack->end() - n_, t_backprop_mode_stack->end());
    }
}

}  // namespace backprop_mode_detail

bool IsBackpropRequired(Context& context) {
    GraphId graph_id = context.default_graph_id();
    return IsBackpropRequired(graph_id);
}

bool IsBackpropRequired(const GraphId& graph_id) {
    backprop_mode_detail::BackpropModeStack* bms = backprop_mode_detail::t_backprop_mode_stack;
    if (bms == nullptr) {
        // No backprop scopes have been created and backprop is thus always required, per default.
        return true;
    }
    auto it = std::find_if(bms->rbegin(), bms->rend(), [&graph_id](const internal::BackpropMode& bm) {
        if (bm.graph_id().has_value()) {
            return graph_id == *bm.graph_id();
        } else {
            // for all graphs in the context
            return &graph_id.context() == &bm.context();
        }
    });
    if (it != bms->rend()) {
        return it->backprop();
    }
    return true;  // Per default.
}

bool IsGradRequired(const Array& array, const nonstd::optional<GraphId>& graph_id) {
    GraphId actual_graph_id = internal::GetArrayGraphId(array, graph_id);
    if (internal::GetArrayBody(array)->HasArrayNode(actual_graph_id)) {
        return IsBackpropRequired(actual_graph_id);
    }
    return false;
}

bool IsGradRequired(const Array& array, AnyGraph /*any_graph*/) {
    const std::vector<std::shared_ptr<internal::ArrayNode>>& array_nodes = internal::GetArrayBody(array)->nodes();
    return std::any_of(array_nodes.begin(), array_nodes.end(), [](const std::shared_ptr<const internal::ArrayNode>& array_node) {
        return IsBackpropRequired(array_node->graph_id());
    });
}

}  // namespace xchainer
