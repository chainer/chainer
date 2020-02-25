#include "chainerx/python/common_export.h"

#include "chainerx/python/check_backward.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/array_body.h"
#include "chainerx/check_backward.h"

#include "chainerx/python/common.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;
using py::literals::operator""_a;

using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;

namespace {

struct ForwardInPython {
    py::object& func;

    std::vector<Array> operator()(const std::vector<Array>& xs_array) const {
        py::gil_scoped_acquire acquire;
        std::vector<ArrayBodyPtr> xs;
        xs.reserve(xs_array.size());
        std::transform(
                xs_array.begin(), xs_array.end(), std::back_inserter(xs), [](Array a) { return internal::MoveArrayBody(std::move(a)); });

        auto ys = py::cast<std::vector<ArrayBodyPtr>>(func(xs));
        return {ys.begin(), ys.end()};
    }
};

}  // namespace

void InitChainerxCheckBackward(pybind11::module& m) {
    m.def("check_backward",
          [](py::object func,
             const std::vector<ArrayBodyPtr>& inputs,
             const std::vector<ArrayBodyPtr>& grad_outputs,
             const std::vector<ArrayBodyPtr>& eps,
             double atol,
             double rtol,
             const absl::optional<BackpropId>& backprop_id) {
              py::gil_scoped_release release;
              CheckBackward(
                      ForwardInPython{func},
                      {inputs.begin(), inputs.end()},
                      {grad_outputs.begin(), grad_outputs.end()},
                      {eps.begin(), eps.end()},
                      2,
                      atol,
                      rtol,
                      backprop_id);
          },
          "func"_a,
          "inputs"_a,
          "grad_outputs"_a,
          "eps"_a,
          "atol"_a = 1e-5,
          "rtol"_a = 1e-4,
          "backprop_id"_a = nullptr);

    m.def("check_double_backward",
          [](py::object func,
             const std::vector<ArrayBodyPtr>& inputs,
             const std::vector<ArrayBodyPtr>& grad_outputs,
             const std::vector<ArrayBodyPtr>& grad_grad_inputs,
             const std::vector<ArrayBodyPtr>& eps,
             double atol,
             double rtol,
             const absl::optional<BackpropId>& backprop_id) {
              py::gil_scoped_release release;
              CheckDoubleBackwardComputation(
                      ForwardInPython{func},
                      {inputs.begin(), inputs.end()},
                      {grad_outputs.begin(), grad_outputs.end()},
                      {grad_grad_inputs.begin(), grad_grad_inputs.end()},
                      {eps.begin(), eps.end()},
                      2,
                      atol,
                      rtol,
                      backprop_id);
          },
          "func"_a,
          "inputs"_a,
          "grad_outputs"_a,
          "grad_grad_inputs"_a,
          "eps"_a,
          "atol"_a = 1e-5,
          "rtol"_a = 1e-4,
          "backprop_id"_a = nullptr);
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
