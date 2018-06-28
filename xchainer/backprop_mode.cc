#include "xchainer/backprop_mode.h"

#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/graph.h"

namespace xchainer {
namespace backprop_mode_detail {

thread_local BackpropModeStack* t_backprop_mode_stack{nullptr};

template <bool kModeFlag>
BackpropModeScope<kModeFlag>::BackpropModeScope(nonstd::optional<GraphId> graph_id) {
    // The outer-most scope creates an instance of BackpropModeStack.
    if (t_backprop_mode_stack == nullptr) {
        t_backprop_mode_stack = new BackpropModeStack{};
    }
    t_backprop_mode_stack->emplace_back(GetDefaultContext(), std::move(graph_id), kModeFlag);
}

template <bool kModeFlag>
BackpropModeScope<kModeFlag>::~BackpropModeScope() {
    assert(t_backprop_mode_stack != nullptr);
    t_backprop_mode_stack->pop_back();
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
