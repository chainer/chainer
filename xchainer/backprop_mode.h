#pragma once

#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

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
    Context& context_;
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
    BackpropModeScope() : BackpropModeScope{nonstd::nullopt} {}

    // No backprop mode for specified graph
    explicit BackpropModeScope(const GraphId& graph_id) : BackpropModeScope{std::move(nonstd::optional<GraphId>{graph_id})} {}

    BackpropModeScope(const BackpropModeScope&) = delete;
    BackpropModeScope(BackpropModeScope&& other) = delete;
    BackpropModeScope& operator=(const BackpropModeScope&) = delete;
    BackpropModeScope& operator=(BackpropModeScope&& other) = delete;

    ~BackpropModeScope();

private:
    explicit BackpropModeScope(nonstd::optional<GraphId> graph_id);
};

template class BackpropModeScope<true>;
template class BackpropModeScope<false>;

}  // namespace backprop_mode_detail

using NoBackpropModeScope = backprop_mode_detail::BackpropModeScope<false>;
using ForceBackpropModeScope = backprop_mode_detail::BackpropModeScope<true>;

namespace internal {

// For test
backprop_mode_detail::BackpropModeStack* GetBackpropModeStack();

}  // namespace internal
}  // namespace xchainer
