#pragma once

#include <string>
#include <utility>

#include "xchainer/context.h"
#include "xchainer/graph.h"

namespace xchainer {

class GraphScope {
public:
    GraphScope(std::string graph_name, Context& context = GetDefaultContext())
        : graph_id_{context.MakeNextGraphId(std::move(graph_name))} {}

    GraphScope(const GraphScope&) = delete;
    GraphScope& operator=(const GraphScope&) = delete;
    GraphScope& operator=(GraphScope&&) = delete;
    GraphScope(GraphScope&& other) = delete;

    ~GraphScope() { graph_id_.context().ReleaseGraphId(graph_id_); }

    GraphId graph_id() const { return graph_id_; }

private:
    GraphId graph_id_;
};

}  // namespace xchainer
