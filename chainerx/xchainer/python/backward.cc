#include "xchainer/python/backward.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/backward.h"
#include "xchainer/graph.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace python_internal {

namespace py = pybind11;

using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;

void InitXchainerBackward(pybind11::module& m) {
    m.def("backward",
          [](const ArrayBodyPtr& body, const nonstd::optional<BackpropId>& backprop_id, bool enable_double_backprop) {
              Array array{body};
              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward(array, backprop_id, double_backprop);
          },
          py::arg(),
          py::arg("backprop_id") = nullptr,
          py::arg("enable_double_backprop") = false);

    m.def("backward",
          [](const std::vector<ArrayBodyPtr>& outputs, const nonstd::optional<BackpropId>& backprop_id, bool enable_double_backprop) {
              std::vector<Array> arrays;
              arrays.reserve(outputs.size());
              std::transform(outputs.begin(), outputs.end(), std::back_inserter(arrays), [](ArrayBodyPtr body) { return Array{body}; });

              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward({arrays.begin(), arrays.end()}, backprop_id, double_backprop);
          },
          py::arg(),
          py::arg("backprop_id") = nullptr,
          py::arg("enable_double_backprop") = false);
}

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
