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
BackpropModeScope<kModeFlag>::BackpropModeScope(const std::vector<BackpropId>& backprop_ids) : n_{backprop_ids.size()} {
    // Need to throw before initializing because thowing error at ctor does not call the dtor.
    for (const BackpropId& backprop_id : backprop_ids) {
        if (&backprop_ids.front().context() != &backprop_id.context()) {
            throw ContextError{"Cannot specify backprop ids with different contexts together."};
        }
    }
    InitializeBackpropModeStack();
    for (const BackpropId& backprop_id : backprop_ids) {
        t_backprop_mode_stack->emplace_back(backprop_id, kModeFlag);
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
    BackpropId backprop_id = context.default_backprop_id();
    return IsBackpropRequired(backprop_id);
}

bool IsBackpropRequired(const BackpropId& backprop_id) {
    backprop_mode_detail::BackpropModeStack* bms = backprop_mode_detail::t_backprop_mode_stack;
    if (bms == nullptr) {
        // No backprop scopes have been created and backprop is thus always required, per default.
        return true;
    }
    auto it = std::find_if(bms->rbegin(), bms->rend(), [&backprop_id](const internal::BackpropMode& bm) {
        if (bm.backprop_id().has_value()) {
            return backprop_id == *bm.backprop_id();
        }
        // for all graphs in the context
        return &backprop_id.context() == &bm.context();
    });
    if (it != bms->rend()) {
        return it->backprop();
    }
    return true;  // Per default.
}

}  // namespace xchainer
