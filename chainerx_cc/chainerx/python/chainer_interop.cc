#include "chainerx/python/chainer_interop.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/array_body.h"
#include "chainerx/array_fwd.h"
#include "chainerx/backward.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"

#include "chainerx/python/common.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;

void InitChainerxChainerInterop(pybind11::module& m) {
    m.def("_function_node_forward",
          [](py::handle function_node, const std::vector<ArrayBodyPtr>& inputs, const std::vector<ArrayBodyPtr>& outputs) {
              CHAINERX_ASSERT(std::all_of(
                      outputs.begin(), outputs.end(), [](const ArrayBodyPtr& array_body) { return array_body->nodes().empty(); }));

              // Prepare input arrays for BackwardBuilder
              std::vector<Array> input_arrays;
              std::vector<ConstArrayRef> input_array_refs;
              input_arrays.reserve(inputs.size());
              input_array_refs.reserve(inputs.size());
              std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_arrays), [](const ArrayBodyPtr& array_body) {
                  return Array{array_body};
              });
              std::transform(input_arrays.begin(), input_arrays.end(), std::back_inserter(input_array_refs), [](const Array& array) {
                  return ConstArrayRef{array};
              });

              // Prepare output arrays for BackwardBuilder
              std::vector<Array> output_arrays;
              std::vector<ConstArrayRef> output_array_refs;
              output_arrays.reserve(outputs.size());
              output_array_refs.reserve(outputs.size());
              std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_arrays), [](const ArrayBodyPtr& array_body) {
                  return Array{array_body};
              });
              std::transform(output_arrays.begin(), output_arrays.end(), std::back_inserter(output_array_refs), [](const Array& array) {
                  return ConstArrayRef{array};
              });

              // Insert backward function
              BackwardBuilder bb{"chainer_function", std::move(input_array_refs), std::move(output_array_refs)};
              if (BackwardBuilder::Target bt = bb.CreateTarget()) {
                  bt.Define([function_node](BackwardContext& bctx) {
                      // Target input indexes
                      std::vector<size_t> target_input_indexes;
                      target_input_indexes.reserve(bctx.input_count());

                      for (size_t i_in = 0; i_in < bctx.input_count(); ++i_in) {
                          if (bctx.is_input_grad_required(i_in)) {
                              target_input_indexes.emplace_back(i_in);
                          }
                      }

                      // Collect incoming output gradients
                      std::vector<ArrayBodyPtr> grad_outputs;
                      for (size_t i_out = 0; i_out < bctx.output_count(); ++i_out) {
                          grad_outputs.emplace_back(internal::GetArrayBody(bctx.output_grad(i_out)));
                      }

                      // Call FunctionNode._backward_chainerx()
                      std::vector<ArrayBodyPtr> grad_inputs;
                      {
                          py::gil_scoped_acquire acquire;
                          py::object func_backward = function_node.attr("_backward_chainerx");
                          py::object py_grad_inputs = func_backward(target_input_indexes, grad_outputs);
                          grad_inputs = py::cast<std::vector<ArrayBodyPtr>>(py_grad_inputs);
                      }
                      CHAINERX_ASSERT(grad_inputs.size() == target_input_indexes.size());

                      // Store computed input gradients
                      for (size_t i = 0; i < target_input_indexes.size(); ++i) {
                          size_t i_in = gsl::at(target_input_indexes, i);
                          ArrayBodyPtr& gx = gsl::at(grad_inputs, i);

                          bctx.input_grad(i_in) = Array{gx};
                      }
                  });
              }
              bb.Finalize();
          },
          py::arg("function_node"),
          py::arg("inputs"),
          py::arg("outputs"));
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
