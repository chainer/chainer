#include "xchainer/backprop_mode.h"

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
    }

    if (graph_ids.has_value()) {
        n_ = graph_ids->size();
        for (GraphId& graph_id : *graph_ids) {
            t_backprop_mode_stack->emplace_back(GetDefaultContext(), std::move(graph_id), kModeFlag);
        }
    } else {
        n_ = 1;
        t_backprop_mode_stack->emplace_back(GetDefaultContext(), nonstd::nullopt, kModeFlag);
    }
}

template <bool kModeFlag>
BackpropModeScope<kModeFlag>::~BackpropModeScope() {
    assert(t_backprop_mode_stack != nullptr);
    assert(t_backprop_mode_stack->size() >= n_);

    for (size_t i = 0; i < n_; ++i) {
        t_backprop_mode_stack->pop_back();
    }

    // Recover thread local variable to nullptr on exiting from the outer-most scope.
    if (t_backprop_mode_stack->empty()) {
        delete t_backprop_mode_stack;
        t_backprop_mode_stack = nullptr;
    }
}

}  // namespace backprop_mode_detail

namespace internal {

backprop_mode_detail::BackpropModeStack* GetBackpropModeStack() { return backprop_mode_detail::t_backprop_mode_stack; }

}  // namespace internal
}  // namespace xchainer
