#pragma once

#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
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
    BackpropModeScope(Context& context = GetDefaultContext()) { BackpropModeScopeImpl(nonstd::nullopt, context); }

    // Backprop mode for specified graphs
    explicit BackpropModeScope(std::vector<GraphId> graph_ids, Context& context = GetDefaultContext()) {
        BackpropModeScopeImpl(std::move(graph_ids), context);
    }

    // Backprop mode for specified graphs
    explicit BackpropModeScope(std::initializer_list<GraphId> graph_ids, Context& context = GetDefaultContext()) {
        BackpropModeScopeImpl(std::vector<GraphId>{graph_ids.begin(), graph_ids.end()}, context);
    }

    BackpropModeScope(const BackpropModeScope&) = delete;
    BackpropModeScope(BackpropModeScope&& other) = delete;
    BackpropModeScope& operator=(const BackpropModeScope&) = delete;
    BackpropModeScope& operator=(BackpropModeScope&& other) = delete;

    ~BackpropModeScope();

private:
    void BackpropModeScopeImpl(nonstd::optional<std::vector<GraphId>> graph_ids, Context& context);

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

// Returns whether the array needs to backprop.
// This takes into account NoBackpropModeScope and ForceBackpropModeScope.
inline bool IsBackpropRequired(const Array& array) {
    const std::vector<std::shared_ptr<ArrayNode>>& array_nodes = array.nodes();
    return std::any_of(array_nodes.begin(), array_nodes.end(), [&array](const std::shared_ptr<const ArrayNode>& array_node) {
        return IsBackpropRequired(array_node->graph_id(), array.device().context());
    });
}

}  // namespace xchainer
