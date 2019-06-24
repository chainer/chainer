#include "chainerx/python/backward.h"

#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/array_body.h"
#include "chainerx/backward.h"
#include "chainerx/graph.h"

#include "chainerx/python/common.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;

namespace {

// Converts a vector of ArrayBody pointers to a vector of Arrays.
std::vector<Array> ConvertToArrays(const std::vector<ArrayBodyPtr>& array_body_ptrs) {
    std::vector<Array> arrays;
    arrays.reserve(array_body_ptrs.size());
    for (const ArrayBodyPtr& body : array_body_ptrs) {
        arrays.emplace_back(body);
    }
    return arrays;
}

}  // namespace

void InitChainerxBackward(pybind11::module& m) {
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
              std::vector<Array> arrays = ConvertToArrays(outputs);
              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward({arrays.begin(), arrays.end()}, backprop_id, double_backprop);
          },
          py::arg(),
          py::arg("backprop_id") = nullptr,
          py::arg("enable_double_backprop") = false);

    m.def("grad",
          [](const std::vector<ArrayBodyPtr>& outputs,
             const std::vector<ArrayBodyPtr>& inputs,
             const nonstd::optional<BackpropId>& backprop_id,
             bool enable_double_backprop,
             const std::vector<ArrayBodyPtr>& grad_inputs,
             const std::vector<ArrayBodyPtr>& grad_outputs){
              std::vector<Array> output_arrays = ConvertToArrays(outputs);
              std::vector<Array> input_arrays = ConvertToArrays(inputs);

              std::vector<Array> grad_input_arrays = ConvertToArrays(grad_inputs);
              std::vector<Array> grad_output_arrays = ConvertToArrays(grad_outputs);

              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              std::vector<nonstd::optional<Array>> grads =
                      Grad({output_arrays.begin(), output_arrays.end()},
                           {input_arrays.begin(), input_arrays.end()},
                           backprop_id,
                           double_backprop,
                           std::vector<ConstArrayRef>{grad_input_arrays.begin(), grad_input_arrays.end()},
                           std::vector<ConstArrayRef>{grad_output_arrays.begin(), grad_output_arrays.end()});
                           //out_grads});
              return internal::MoveArrayBodies(std::move(grads));
          },
          py::arg(),  // outputs
          py::arg(),  // inputs
          py::arg("backprop_id") = nullptr,
          py::arg("enable_double_backprop") = false,
          py::arg("grad_inputs") = std::vector<ArrayBodyPtr>{},
          py::arg("grad_outputs") = std::vector<ArrayBodyPtr>{});
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
