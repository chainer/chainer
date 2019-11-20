#include "chainerx/backprop_mode.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/array_node.h"
#include "chainerx/constant.h"
#include "chainerx/context.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/thread_local_state.h"

namespace chainerx {
namespace {

using internal::BackpropModeStack;

}  // namespace

namespace backprop_mode_detail {

template <bool kModeFlag>
BackpropModeScope<kModeFlag>::BackpropModeScope(Context& context) : n_{1} {
    BackpropModeStack& backprop_mode_stack = internal::GetInternalThreadLocalState().backprop_mode_stack;
    backprop_mode_stack.emplace_back(context, kModeFlag);
}

template <bool kModeFlag>
BackpropModeScope<kModeFlag>::BackpropModeScope(const std::vector<BackpropId>& backprop_ids) : n_{backprop_ids.size()} {
    // Need to throw before initializing because thowing error at ctor does not call the dtor.
    for (const BackpropId& backprop_id : backprop_ids) {
        if (&backprop_ids.front().context() != &backprop_id.context()) {
            throw ContextError{"Cannot specify backprop ids with different contexts together."};
        }
    }
    BackpropModeStack& backprop_mode_stack = internal::GetInternalThreadLocalState().backprop_mode_stack;
    for (const BackpropId& backprop_id : backprop_ids) {
        backprop_mode_stack.emplace_back(backprop_id, kModeFlag);
    }
}

template <bool kModeFlag>
BackpropModeScope<kModeFlag>::~BackpropModeScope() {
    BackpropModeStack& backprop_mode_stack = internal::GetInternalThreadLocalState().backprop_mode_stack;
    CHAINERX_ASSERT(backprop_mode_stack.size() >= n_);

    backprop_mode_stack.erase(backprop_mode_stack.end() - n_, backprop_mode_stack.end());
}

template class BackpropModeScope<true>;
template class BackpropModeScope<false>;

}  // namespace backprop_mode_detail

bool IsBackpropRequired(Context& context) {
    BackpropId backprop_id = context.default_backprop_id();
    return IsBackpropRequired(backprop_id);
}

bool IsBackpropRequired(const BackpropId& backprop_id) {
    BackpropModeStack& bms = internal::GetInternalThreadLocalState().backprop_mode_stack;
    auto it = std::find_if(bms.rbegin(), bms.rend(), [&backprop_id](const internal::BackpropMode& bm) {
        if (bm.backprop_id().has_value()) {
            return backprop_id == *bm.backprop_id();
        }
        // for all graphs in the context
        return &backprop_id.context() == &bm.context();
    });
    if (it != bms.rend()) {
        return it->backprop();
    }
    return true;  // Per default.
}

}  // namespace chainerx
