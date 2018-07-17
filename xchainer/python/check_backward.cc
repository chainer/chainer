#include "xchainer/python/check_backward.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/check_backward.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace python_internal {

namespace py = pybind11;

using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;

namespace {

struct ForwardInPython {
    py::object& func;

    std::vector<Array> operator()(const std::vector<Array>& xs_array) const {
        std::vector<ArrayBodyPtr> xs;
        xs.reserve(xs_array.size());
        std::transform(xs_array.begin(), xs_array.end(), std::back_inserter(xs), [](Array a) { return a.move_body(); });

        auto ys = py::cast<std::vector<ArrayBodyPtr>>(func(xs));
        return {ys.begin(), ys.end()};
    }
};

}  // namespace

void InitXchainerCheckBackward(pybind11::module& m) {
    m.def("check_backward",
          [](py::object func,
             const std::vector<ArrayBodyPtr>& inputs,
             const std::vector<ArrayBodyPtr>& grad_outputs,
             const std::vector<ArrayBodyPtr>& eps,
             double atol,
             double rtol,
             const GraphId& graph_id) {
              CheckBackward(
                      ForwardInPython{func},
                      {inputs.begin(), inputs.end()},
                      {grad_outputs.begin(), grad_outputs.end()},
                      {eps.begin(), eps.end()},
                      atol,
                      rtol,
                      graph_id);
          },
          py::arg("func"),
          py::arg("inputs"),
          py::arg("grad_outputs"),
          py::arg("eps"),
          py::arg("atol") = 1e-5,
          py::arg("rtol") = 1e-4,
          py::arg("graph_id") = kDefaultGraphId);

    m.def("check_double_backward",
          [](py::object func,
             const std::vector<ArrayBodyPtr>& inputs,
             const std::vector<ArrayBodyPtr>& grad_outputs,
             const std::vector<ArrayBodyPtr>& grad_grad_inputs,
             const std::vector<ArrayBodyPtr>& eps,
             double atol,
             double rtol,
             const GraphId& graph_id) {
              CheckDoubleBackwardComputation(
                      ForwardInPython{func},
                      {inputs.begin(), inputs.end()},
                      {grad_outputs.begin(), grad_outputs.end()},
                      {grad_grad_inputs.begin(), grad_grad_inputs.end()},
                      {eps.begin(), eps.end()},
                      atol,
                      rtol,
                      graph_id);
          },
          py::arg("func"),
          py::arg("inputs"),
          py::arg("grad_outputs"),
          py::arg("grad_grad_inputs"),
          py::arg("eps"),
          py::arg("atol") = 1e-5,
          py::arg("rtol") = 1e-4,
          py::arg("graph_id") = kDefaultGraphId);
}

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
