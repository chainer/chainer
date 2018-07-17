#include "xchainer/python/backprop_mode.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/backprop_mode.h"
#include "xchainer/context.h"
#include "xchainer/graph.h"

#include "xchainer/python/common.h"
#include "xchainer/python/context.h"

namespace xchainer {
namespace python {
namespace python_internal {
namespace {

namespace py = pybind11;  // standard convention

template <class BackpropModeScope>
class PyBackpropModeScope {
public:
    PyBackpropModeScope() = default;
    explicit PyBackpropModeScope(std::vector<GraphId> graph_ids) : graph_ids_{std::move(graph_ids)} {}

    void Enter() {
        scope_ = graph_ids_.has_value() ? std::make_unique<BackpropModeScope>(*graph_ids_) : std::make_unique<BackpropModeScope>();
    }
    void Exit(py::args args) {
        (void)args;  // unused
        scope_.reset();
    }

private:
    // optional requires having copy ctor, so use unique_ptr instead
    std::unique_ptr<BackpropModeScope> scope_;
    nonstd::optional<std::vector<GraphId>> graph_ids_{};
};

template <class BackpropModeScope>
void InitXchainerBackpropModeScope(pybind11::module& m, const char* class_name, const char* function_name) {
    py::class_<PyBackpropModeScope<BackpropModeScope>> c{m, class_name};
    c.def("__enter__", &PyBackpropModeScope<BackpropModeScope>::Enter);
    c.def("__exit__", &PyBackpropModeScope<BackpropModeScope>::Exit);

    m.def(function_name, []() { return PyBackpropModeScope<BackpropModeScope>{}; });
    m.def(function_name, [](GraphId graph_id) { return PyBackpropModeScope<BackpropModeScope>{{std::move(graph_id)}}; });
    m.def(function_name, [](const std::vector<GraphId>& graph_ids) { return PyBackpropModeScope<BackpropModeScope>{graph_ids}; });
}

}  // namespace

void InitXchainerBackpropMode(pybind11::module& m) {
    InitXchainerBackpropModeScope<NoBackpropModeScope>(m, "NoBackpropMode", "no_backprop_mode");
    InitXchainerBackpropModeScope<ForceBackpropModeScope>(m, "ForceBackpropMode", "force_backprop_mode");

    m.def("is_backprop_required",
          [](const GraphId& graph_id, py::handle context) { return IsBackpropRequired(graph_id, GetContext(context)); },
          py::arg("graph_id") = kDefaultGraphId,
          py::arg("context") = py::none());
}

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
