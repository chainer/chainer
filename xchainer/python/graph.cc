#include "xchainer/python/graph.h"

#include "xchainer/graph.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

void InitXchainerGraph(pybind11::module& m) {
    py::class_<AnyGraph>{m, "AnyGraph"};  // NOLINT: misc-unused-raii

    // TODO(imanishi): Add module function to retrieve default graph id.
    m.attr("anygraph") = AnyGraph{};
}

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
