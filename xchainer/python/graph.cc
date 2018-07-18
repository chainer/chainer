#include "xchainer/python/graph.h"

#include <memory>
#include <string>
#include <utility>

#include "xchainer/context.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/graph_scope.h"

#include "xchainer/python/common.h"
#include "xchainer/python/context.h"

namespace xchainer {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

class PyGraphScope {
public:
    explicit PyGraphScope(std::string graph_name, Context& context) : graph_name_{std::move(graph_name)}, context_{context} {}
    void Enter() {
        if (scope_ != nullptr) {
            throw XchainerError{"Graph scope cannot be nested."};
        }
        if (exited_) {
            throw XchainerError{"Exited graph scope cannot be reused."};
        }
        scope_ = std::make_unique<GraphScope>(graph_name_, context_);
    }
    void Exit(py::args args) {
        (void)args;  // unused
        exited_ = true;
        scope_.reset();
    }

private:
    std::string graph_name_;
    Context& context_;
    std::unique_ptr<GraphScope> scope_;
    bool exited_{false};
};

void InitXchainerGraph(pybind11::module& m) {
    py::class_<AnyGraph>{m, "AnyGraph"};  // NOLINT: misc-unused-raii

    // TODO(imanishi): Add module function to retrieve default graph id.
    m.attr("anygraph") = AnyGraph{};
}

void InitXchainerGraphScope(pybind11::module& m) {
    py::class_<PyGraphScope> c{m, "GraphScope"};
    c.def("__enter__", &PyGraphScope::Enter);
    c.def("__exit__", &PyGraphScope::Exit);

    m.def("graph_scope",
          [](const std::string& graph_name, py::handle context) { return PyGraphScope(graph_name, GetContext(context)); },
          py::arg("graph_name"),
          py::arg("context") = py::none());
}

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
