#include "chainerx/python/common_export.h"

#include "chainerx/python/backward.h"

#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/array_body.h"
#include "chainerx/backward.h"
#include "chainerx/graph.h"

#include "chainerx/python/common.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;
using py::literals::operator""_a;

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
          [](const ArrayBodyPtr& body,
             const absl::optional<BackpropId>& backprop_id,
             bool enable_double_backprop,
             absl::optional<float> loss_scale) {
              Array array{body};
              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward(array, backprop_id, double_backprop, loss_scale);
          },
          py::arg(),
          "backprop_id"_a = nullptr,
          "enable_double_backprop"_a = false,
          "loss_scale"_a = nullptr);

    m.def("backward",
          [](const std::vector<ArrayBodyPtr>& outputs,
             const absl::optional<BackpropId>& backprop_id,
             bool enable_double_backprop,
             absl::optional<float> loss_scale) {
              std::vector<Array> arrays = ConvertToArrays(outputs);
              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward({arrays.begin(), arrays.end()}, backprop_id, double_backprop, loss_scale);
          },
          py::arg(),
          "backprop_id"_a = nullptr,
          "enable_double_backprop"_a = false,
          "loss_scale"_a = nullptr);

    m.def("grad",
          [](const std::vector<ArrayBodyPtr>& outputs,
             const std::vector<ArrayBodyPtr>& inputs,
             const absl::optional<BackpropId>& backprop_id,
             bool enable_double_backprop,
             bool set_grad,
             bool retain_grad,
             const std::vector<ArrayBodyPtr>& grad_outputs,
             absl::optional<float> loss_scale) {
              std::vector<Array> output_arrays = ConvertToArrays(outputs);
              std::vector<Array> input_arrays = ConvertToArrays(inputs);

              std::vector<Array> grad_output_arrays = ConvertToArrays(grad_outputs);

              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              std::vector<absl::optional<Array>> grads =
                      Grad({output_arrays.begin(), output_arrays.end()},
                           {input_arrays.begin(), input_arrays.end()},
                           backprop_id,
                           double_backprop,
                           set_grad,
                           retain_grad,
                           std::vector<ConstArrayRef>{grad_output_arrays.begin(), grad_output_arrays.end()},
                           loss_scale);
              return internal::MoveArrayBodies(std::move(grads));
          },
          py::arg(),  // outputs
          py::arg(),  // inputs
          "backprop_id"_a = nullptr,
          "enable_double_backprop"_a = false,
          "set_grad"_a = false,
          "retain_grad"_a = false,
          "grad_outputs"_a = std::vector<ArrayBodyPtr>{},
          "loss_scale"_a = nullptr);
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
