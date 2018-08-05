#pragma once

#include <string>
#include <utility>

#include "xchainer/context.h"
#include "xchainer/graph.h"

namespace xchainer {

class BackpropScope {
public:
    explicit BackpropScope(std::string graph_name, Context& context = GetDefaultContext())
        : graph_id_{context.MakeNextGraphId(std::move(graph_name))} {}

    BackpropScope(const BackpropScope&) = delete;
    BackpropScope& operator=(const BackpropScope&) = delete;
    BackpropScope& operator=(BackpropScope&&) = delete;
    BackpropScope(BackpropScope&& other) = delete;

    ~BackpropScope() { graph_id_.context().ReleaseGraphId(graph_id_); }

    GraphId graph_id() const { return graph_id_; }

private:
    GraphId graph_id_;
};

}  // namespace xchainer
