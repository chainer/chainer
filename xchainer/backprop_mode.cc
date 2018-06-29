#include "xchainer/backprop_mode.h"

#include <algorithm>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/graph.h"

namespace xchainer {
namespace backprop_mode_detail {

thread_local BackpropModeStack* t_backprop_mode_stack{nullptr};

template <bool kModeFlag>
void BackpropModeScope<kModeFlag>::BackpropModeScopeImpl(nonstd::optional<std::vector<GraphId>> graph_ids) {
    // The outer-most scope creates an instance of BackpropModeStack.
    if (t_backprop_mode_stack == nullptr) {
        t_backprop_mode_stack = new BackpropModeStack{};
        is_outermost_ = true;
    }

    Context& context = GetDefaultContext();
    if (graph_ids.has_value()) {
        n_ = graph_ids->size();
        for (GraphId& graph_id : *graph_ids) {
            t_backprop_mode_stack->emplace_back(context, std::move(graph_id), kModeFlag);
        }
    } else {
        n_ = 1;
        t_backprop_mode_stack->emplace_back(context, nonstd::nullopt, kModeFlag);
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

bool IsBackpropRequired(const GraphId& graph_id, Context& context) {
    backprop_mode_detail::BackpropModeStack* bms = backprop_mode_detail::t_backprop_mode_stack;
    if (bms == nullptr) {
        // No backprop scopes have been created and backprop is thus always required, per default.
        return true;
    }
    auto it = std::find_if(bms->rbegin(), bms->rend(), [&graph_id, &context](const internal::BackpropMode& bm) {
        return &context == &bm.context() && (!bm.graph_id().has_value() || graph_id == *bm.graph_id());
    });
    if (it != bms->rend()) {
        return it->backprop();
    }
    return true;  // Per default.
}

}  // namespace xchainer
