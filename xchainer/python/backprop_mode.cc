#include "xchainer/python/backprop_mode.h"

#include <memory>
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

template <class PyBackpropModeScope>
void InitXchainerBackpropModeScope(pybind11::module& m, const std::string& class_name, const std::string& function_name) {
    py::class_<PyBackpropModeScope> c{m, class_name.c_str()};
    c.def("__enter__", &PyBackpropModeScope::Enter);
    c.def("__exit__", &PyBackpropModeScope::Exit);

    m.def(function_name.c_str(), []() { return PyBackpropModeScope{}; });
    m.def(function_name.c_str(), [](GraphId graph_id) { return PyBackpropModeScope{{std::move(graph_id)}}; });
    m.def(function_name.c_str(), [](const std::vector<GraphId>& graph_ids) { return PyBackpropModeScope{graph_ids}; });
}

}  // namespace

void InitXchainerBackpropMode(pybind11::module& m) {
    InitXchainerBackpropModeScope<PyNoBackpropModeScope>(m, "NoBackpropMode", "no_backprop_mode");
    InitXchainerBackpropModeScope<PyForceBackpropModeScope>(m, "ForceBackpropMode", "force_backprop_mode");

    m.def("is_backprop_required",
          [](const GraphId& graph_id, py::handle context) { return IsBackpropRequired(graph_id, GetContext(context)); },
          py::arg("graph_id") = kDefaultGraphId,
          py::arg("context") = py::none());
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
