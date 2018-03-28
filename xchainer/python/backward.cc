#include "xchainer/python/backward.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/backward.h"
#include "xchainer/graph.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

using ArrayBodyPtr = std::shared_ptr<xchainer::internal::ArrayBody>;

void InitXchainerBackward(pybind11::module& m) {
    m.def("backward",
          [](const ArrayBodyPtr& body, const GraphId& graph_id, bool enable_double_backprop) {
              Array array{body};
              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward(array, graph_id, double_backprop);
          },
          py::arg(),
          py::arg("graph_id") = kDefaultGraphId,
          py::arg("enable_double_backprop") = false);

    m.def("backward",
          [](const std::vector<ArrayBodyPtr>& outputs, const GraphId& graph_id, bool enable_double_backprop) {
              std::vector<Array> arrays;
              arrays.reserve(outputs.size());
              std::transform(outputs.begin(), outputs.end(), std::back_inserter(arrays), [](ArrayBodyPtr body) { return Array{body}; });

              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward({arrays.begin(), arrays.end()}, graph_id, double_backprop);
          },
          py::arg(),
          py::arg("graph_id") = kDefaultGraphId,
          py::arg("enable_double_backprop") = false);
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
