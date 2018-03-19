#include "xchainer/python/backward.h"

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/backward.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;
using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;

void InitXchainerBackward(pybind11::module& m) {
    m.def("backward",
          [](const ArrayBodyPtr& body, const GraphId& graph_id, bool enable_double_backprop) {
              Array array{body};
              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward(array, graph_id, double_backprop);
          },
          py::arg().noconvert(),
          py::arg("graph_id") = kDefaultGraphId,
          py::arg("enable_double_backprop") = false);
}

}  // namespace xchainer

