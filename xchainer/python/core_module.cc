#include <pybind11/pybind11.h>

#include "xchainer/array.h"
#include "xchainer/backprop.h"
#include "xchainer/constant.h"
#include "xchainer/python/array.h"
#include "xchainer/python/backend.h"
#include "xchainer/python/common.h"
#include "xchainer/python/context.h"
#include "xchainer/python/device.h"
#include "xchainer/python/dtype.h"
#include "xchainer/python/error.h"
#include "xchainer/python/scalar.h"
#include "xchainer/python/shape.h"

namespace xchainer {

namespace py = pybind11;

namespace {

void InitXchainerModule(pybind11::module& m) {
    using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;

    m.doc() = "xChainer";
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer.core.*"

    m.attr("DEFAULT_GRAPH_ID") = kDefaultGraphId;

    m.def("backward",
          [](const ArrayBodyPtr& body, const GraphId& graph_id, bool enable_double_backprop) {
              Array array{body};
              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward(array, graph_id, double_backprop);
          },
          py::arg().noconvert(), py::arg("graph_id") = kDefaultGraphId, py::arg("enable_double_backprop") = false);
}
}  // namespace
}  // namespace xchainer

PYBIND11_MODULE(_core, m) {  // NOLINT
    xchainer::InitXchainerModule(m);
    xchainer::InitXchainerContext(m);
    xchainer::InitXchainerBackend(m);
    xchainer::InitXchainerDevice(m);
    xchainer::InitXchainerDtype(m);
    xchainer::InitXchainerError(m);
    xchainer::InitXchainerScalar(m);
    xchainer::InitXchainerShape(m);
    xchainer::InitXchainerArray(m);
}
