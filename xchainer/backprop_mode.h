#pragma once

#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace internal {

class BackpropMode {
public:
    BackpropMode(Context& context, const nonstd::optional<GraphId>& graph_id, bool backprop)
        : context_{context}, graph_id_{graph_id}, backprop_{backprop} {}
    BackpropMode(Context& context, const GraphId& graph_id, bool backprop) : context_{context}, graph_id_{graph_id}, backprop_{backprop} {}

    Context& context() const { return context_; }

    const nonstd::optional<GraphId>& graph_id() const { return graph_id_; }

    bool backprop() const { return backprop_; }

private:
    // Using reference_wrapper to make this class move assignable because we use vector::erase in BackpropModeScope dtor.
    std::reference_wrapper<Context> context_;

    nonstd::optional<GraphId> graph_id_;
    bool backprop_;  // false for NoBackpropMode, and true for ForceBackpropMode
};

}  // namespace internal

namespace backprop_mode_detail {

using BackpropModeStack = std::vector<internal::BackpropMode>;

template <bool kModeFlag>
class BackpropModeScope {
public:
    // Backprop mode for all graphs
    BackpropModeScope() { BackpropModeScopeImpl(nonstd::nullopt); }

    // Backprop mode for specified graphs
    explicit BackpropModeScope(std::vector<GraphId> graph_ids) { BackpropModeScopeImpl(std::move(graph_ids)); }

    // Backprop mode for specified graphs
    explicit BackpropModeScope(std::initializer_list<GraphId> graph_ids) {
        BackpropModeScopeImpl(std::vector<GraphId>{graph_ids.begin(), graph_ids.end()});
    }

    BackpropModeScope(const BackpropModeScope&) = delete;
    BackpropModeScope(BackpropModeScope&& other) = delete;
    BackpropModeScope& operator=(const BackpropModeScope&) = delete;
    BackpropModeScope& operator=(BackpropModeScope&& other) = delete;

    ~BackpropModeScope();

private:
    void BackpropModeScopeImpl(nonstd::optional<std::vector<GraphId>> graph_ids);

    // Number of BackpropMode instances pushed to the stack.
    size_t n_{};
    bool is_outermost_{false};
};

template class BackpropModeScope<true>;
template class BackpropModeScope<false>;

}  // namespace backprop_mode_detail

// Make a context which disables back-propagation.
using NoBackpropModeScope = backprop_mode_detail::BackpropModeScope<false>;

// Make a context which enables back-propagation.
using ForceBackpropModeScope = backprop_mode_detail::BackpropModeScope<true>;

bool IsBackpropRequired(const GraphId& graph_id = kDefaultGraphId, Context& context = GetDefaultContext());

}  // namespace xchainer
