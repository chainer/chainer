#include "xchainer/python/backprop_mode.h"

#include <nonstd/optional.hpp>

#include "xchainer/backprop_mode.h"
#include "xchainer/context.h"
#include "xchainer/graph.h"

#include "xchainer/python/common.h"
#include "xchainer/python/context.h"

namespace xchainer {
namespace python {
namespace internal {
namespace {

namespace py = pybind11;  // standard convention

template <class BackpropModeScope>
class PyBackpropModeScope {
public:
    explicit PyBackpropModeScope() {}
    explicit PyBackpropModeScope(std::vector<GraphId> graph_ids) : graph_ids_{std::move(graph_ids)} {}
    void Enter() { scope_ = graph_ids_ ? std::make_unique<BackpropModeScope>(*graph_ids_) : std::make_unique<BackpropModeScope>(); }
    void Exit(py::args args) {
        (void)args;  // unused
        scope_.reset();
    }

private:
    // optional requires having copy ctor, so use unique_ptr instead
    std::unique_ptr<BackpropModeScope> scope_;
    nonstd::optional<std::vector<GraphId>> graph_ids_{};
};

using PyNoBackpropModeScope = PyBackpropModeScope<NoBackpropModeScope>;
using PyForceBackpropModeScope = PyBackpropModeScope<ForceBackpropModeScope>;

void InitXchainerNoBackpropMode(pybind11::module& m) {
    py::class_<PyNoBackpropModeScope> c{m, "NoBackpropMode"};
    c.def("__enter__", &PyNoBackpropModeScope::Enter);
    c.def("__exit__", &PyNoBackpropModeScope::Exit);

    m.def("no_backprop_mode", []() { return PyNoBackpropModeScope{}; });
    m.def("no_backprop_mode", [](GraphId graph_id) { return PyNoBackpropModeScope{{std::move(graph_id)}}; });
    m.def("no_backprop_mode", [](const std::vector<GraphId>& graph_ids) { return PyNoBackpropModeScope{graph_ids}; });
}

void InitXchainerForceBackpropMode(pybind11::module& m) {
    py::class_<PyForceBackpropModeScope> c{m, "ForceBackpropMode"};
    c.def("__enter__", &PyForceBackpropModeScope::Enter);
    c.def("__exit__", &PyForceBackpropModeScope::Exit);

    m.def("force_backprop_mode", []() { return PyForceBackpropModeScope{}; });
    m.def("force_backprop_mode", [](GraphId graph_id) { return PyForceBackpropModeScope{{std::move(graph_id)}}; });
    m.def("force_backprop_mode", [](const std::vector<GraphId>& graph_ids) { return PyForceBackpropModeScope{graph_ids}; });
}
}  // namespace

void InitXchainerBackpropMode(pybind11::module& m) {
    InitXchainerNoBackpropMode(m);
    InitXchainerForceBackpropMode(m);

    m.def("is_backprop_required",
          [](const GraphId& graph_id, py::handle context) { return IsBackpropRequired(graph_id, GetContext(context)); },
          py::arg("graph_id") = kDefaultGraphId,
          py::arg("context") = py::none());
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
