#include "xchainer/python/graph.h"

#include <memory>
#include <string>

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;  // standard convention

void InitXchainerGraph(pybind11::module& m) {
    py::class_<GraphId> c{m, "GraphId"};
    c.def(py::init<GraphId::Type>());
    c.def(py::init<std::string>());
    c.def("__repr__", &GraphId::ToString);

    py::enum_<GraphId::Type>(c, "Type")
            .value("NAMED_GRAPH_ID", GraphId::kNamed)
            .value("DEFAULT_GRAPH_ID", GraphId::kDefault)
            .value("ANY_GRAPH_ID", GraphId::kAny)
            .export_values();

    py::implicitly_convertible<GraphId::Type, GraphId>();
    py::implicitly_convertible<std::string, GraphId>();

    // Aliases.
    m.attr("DEFAULT_GRAPH_ID") = GraphId::kDefault;
    m.attr("ANY_GRAPH_ID") = GraphId::kAny;
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
