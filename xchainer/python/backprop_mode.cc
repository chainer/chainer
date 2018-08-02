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
    explicit PyBackpropModeScope(std::vector<GraphId> graph_ids) : graph_ids_{std::move(graph_ids)} {}
    explicit PyBackpropModeScope(Context& context) : context_{&context} {}

    void Enter() {
        if (graph_ids_.has_value()) {
            scope_ = std::make_unique<BackpropModeScope>(*graph_ids_);
        } else {
            assert(context_ != nullptr);
            scope_ = std::make_unique<BackpropModeScope>(*context_);
        }
    }
    void Exit(py::args args) {
        (void)args;  // unused
        scope_.reset();
    }

private:
    // optional requires having copy ctor, so use unique_ptr instead
    std::unique_ptr<BackpropModeScope> scope_;
    Context* context_{nullptr};
    nonstd::optional<std::vector<GraphId>> graph_ids_{};
};

template <class BackpropModeScope>
void InitXchainerBackpropModeScope(pybind11::module& m, const char* class_name, const char* function_name) {
    py::class_<PyBackpropModeScope<BackpropModeScope>> c{m, class_name};
    c.def("__enter__", &PyBackpropModeScope<BackpropModeScope>::Enter);
    c.def("__exit__", &PyBackpropModeScope<BackpropModeScope>::Exit);

    m.def(function_name, [](const GraphId& graph_id) { return PyBackpropModeScope<BackpropModeScope>{{graph_id}}; });
    m.def(function_name, [](const std::vector<GraphId>& graph_ids) { return PyBackpropModeScope<BackpropModeScope>{graph_ids}; });
    m.def(function_name,
          [](py::object context) { return PyBackpropModeScope<BackpropModeScope>{GetContext(context)}; },
          py::arg("context") = py::none());
}

}  // namespace

void InitXchainerBackpropMode(pybind11::module& m) {
    InitXchainerBackpropModeScope<NoBackpropModeScope>(m, "NoBackpropMode", "no_backprop_mode");
    InitXchainerBackpropModeScope<ForceBackpropModeScope>(m, "ForceBackpropMode", "force_backprop_mode");

    m.def("is_backprop_required", [](const GraphId& graph_id) { return IsBackpropRequired(graph_id); }, py::arg("graph_id"));
    m.def("is_backprop_required",
          [](py::handle context) { return IsBackpropRequired(GetContext(context)); },
          py::arg("context") = py::none());
}

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
