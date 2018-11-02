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
          [](py::object function_node,
             const std::vector<ArrayBodyPtr>& inputs,
             const std::vector<ArrayBodyPtr>& outputs,
             const std::vector<size_t>& input_indexes_to_retain,
             const std::vector<size_t>& output_indexes_to_retain) {
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
                  auto function_node_ptr = std::make_shared<py::object>(std::move(function_node), [](py::object* ptr) {
                      py::gil_scoped_acquire acquire;
                      delete ptr;
                  });

                  std::vector<RetainedInputToken> retained_input_tokens;
                  std::transform(
                          input_indexes_to_retain.begin(),
                          input_indexes_to_retain.end(),
                          std::back_inserter(retained_input_tokens),
                          [&bb](size_t i) { return bb.RetainInput(i); });
                  std::vector<RetainedOutputToken> retained_output_tokens;
                  std::transform(
                          output_indexes_to_retain.begin(),
                          output_indexes_to_retain.end(),
                          std::back_inserter(retained_output_tokens),
                          [&bb](size_t i) { return bb.RetainOutput(i); });

                  bt.Define([function_node_ptr = std::move(function_node_ptr),
                             in_toks = std::move(retained_input_tokens),
                             out_toks = std::move(retained_output_tokens)](BackwardContext& bctx) {
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
                          if (auto gy = bctx.output_grad(i_out)) {
                              grad_outputs.emplace_back(internal::GetArrayBody(*gy));
                          } else {
                              grad_outputs.emplace_back(nullptr);
                          }
                      }

                      // Get retained inputs and outputs
                      std::vector<ArrayBodyPtr> retained_inputs;
                      std::transform(
                              in_toks.begin(), in_toks.end(), std::back_inserter(retained_inputs), [&bctx](const RetainedInputToken& tok) {
                                  return internal::GetArrayBody(bctx.GetRetainedInput(tok));
                              });
                      std::vector<ArrayBodyPtr> retained_outputs;
                      std::transform(
                              out_toks.begin(),
                              out_toks.end(),
                              std::back_inserter(retained_outputs),
                              [&bctx](const RetainedOutputToken& tok) { return internal::GetArrayBody(bctx.GetRetainedOutput(tok)); });

                      // Call FunctionNode._backward_chainerx()
                      std::vector<ArrayBodyPtr> grad_inputs;
                      {
                          py::gil_scoped_acquire acquire;
                          py::object func_backward = function_node_ptr->attr("_backward_chainerx");
                          py::object py_grad_inputs = func_backward(target_input_indexes, grad_outputs, retained_inputs, retained_outputs);
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
          py::arg("outputs"),
          py::arg("input_indexes_to_retain"),
          py::arg("output_indexes_to_retain"));
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
