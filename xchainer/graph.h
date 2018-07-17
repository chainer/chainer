#pragma once

#include <string>

namespace xchainer {

using GraphSubId = uint64_t;

class GraphId {
public:
    GraphId(Context& context, GraphSubId sub_id) : context_{context}, sub_id_{sub_id} {}
    Context& context() const { return context_; }
    GraphSubId sub_id() const { return sub_id_; }

private:
    Context& context_;
    GraphSubId sub_id_;
};

// Used to represent any graph (id).
class AnyGraph {};

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
