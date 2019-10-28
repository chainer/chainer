#pragma once

#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/context.h"
#include "chainerx/graph.h"

namespace chainerx {
namespace internal {

class BackpropMode {
public:
    BackpropMode(Context& context, bool backprop) : context_{context}, backprop_id_{absl::nullopt}, backprop_{backprop} {}
    BackpropMode(const BackpropId& backprop_id, bool backprop)
        : context_{backprop_id.context()}, backprop_id_{backprop_id}, backprop_{backprop} {}

    Context& context() const { return context_; }

    const absl::optional<BackpropId>& backprop_id() const { return backprop_id_; }

    bool backprop() const { return backprop_; }

private:
    // Using reference_wrapper to make this class move assignable because we use vector::erase in BackpropModeScope dtor.
    std::reference_wrapper<Context> context_;

    absl::optional<BackpropId> backprop_id_;
    bool backprop_;  // false for NoBackpropMode, and true for ForceBackpropMode
};

using BackpropModeStack = std::vector<internal::BackpropMode>;

}  // namespace internal

namespace backprop_mode_detail {

template <bool kModeFlag>
class BackpropModeScope {
public:
    // Backprop mode for all graphs in the specified context
    explicit BackpropModeScope(Context& context = GetDefaultContext());

    // Backprop mode for specified graphs
    explicit BackpropModeScope(const std::vector<BackpropId>& backprop_ids);

    // Backprop mode for specified graphs
    BackpropModeScope(std::initializer_list<BackpropId> backprop_ids) : BackpropModeScope{{backprop_ids.begin(), backprop_ids.end()}} {}

    BackpropModeScope(const BackpropModeScope&) = delete;
    BackpropModeScope(BackpropModeScope&& other) = delete;
    BackpropModeScope& operator=(const BackpropModeScope&) = delete;
    BackpropModeScope& operator=(BackpropModeScope&& other) = delete;

    ~BackpropModeScope();

private:
    // Number of BackpropMode instances pushed to the stack.
    size_t n_{};
};

extern template class BackpropModeScope<true>;
extern template class BackpropModeScope<false>;

}  // namespace backprop_mode_detail

// Make a context which disables back-propagation.
using NoBackpropModeScope = backprop_mode_detail::BackpropModeScope<false>;

// Make a context which enables back-propagation.
using ForceBackpropModeScope = backprop_mode_detail::BackpropModeScope<true>;

bool IsBackpropRequired(Context& context = GetDefaultContext());
bool IsBackpropRequired(const BackpropId& backprop_id);

}  // namespace chainerx
