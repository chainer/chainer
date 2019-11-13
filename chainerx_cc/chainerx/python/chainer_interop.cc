#include "chainerx/python/common_export.h"

#include "chainerx/python/chainer_interop.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <absl/types/optional.h>
#include <gsl/gsl>

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
using py::literals::operator""_a;

using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;

namespace {

inline bool IsUniqueAndIncreasingIndexes(const std::vector<size_t>& vec, size_t upper) {
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0 && vec[i] <= vec[i - 1]) {
            return false;
        }
        if (upper <= vec[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

void InitChainerxChainerInterop(pybind11::module& m) {
    m.def("_function_node_forward",
          [](py::object function_node,
             const std::vector<ArrayBodyPtr>& inputs,
             const std::vector<ArrayBodyPtr>& outputs,
             const std::vector<size_t>& input_indexes_to_retain,
             const std::vector<size_t>& output_indexes_to_retain) {
              // Implementation note:
              // There are two kinds of indices:
              // o: Original indices. This is the indices used in Python world. It includes None arrays.
              // r: Reduced indices. This is the indices that ChainerX C++ impl handles. None arrays are omitted.
              // o and r are used as abbreviations of these.
              // i and j are used as variables of respective kinds of indices.
              CHAINERX_ASSERT(IsUniqueAndIncreasingIndexes(input_indexes_to_retain, inputs.size()));
              CHAINERX_ASSERT(IsUniqueAndIncreasingIndexes(output_indexes_to_retain, outputs.size()));
              CHAINERX_ASSERT(std::all_of(outputs.begin(), outputs.end(), [](const ArrayBodyPtr& array_body) {
                  return array_body == nullptr || array_body->nodes().empty();
              }));
              const size_t chainer_output_count = outputs.size();

              // Prepare input/output arrays for BackwardBuilder
              // {in,out}put_index_map maps original input/output indices to reduced indices where None outputs are omitted

              auto get_reduced_arrays = [](const std::vector<ArrayBodyPtr>& array_bodies) {
                  // Given the input/output array bodies, construct an index mapping between original <o> indices and
                  // reduced <r> indices where None arrays are omitted.
                  std::vector<Array> reduced_arrays;
                  std::vector<size_t> index_r2o_map;
                  std::vector<absl::optional<size_t>> index_o2r_map(array_bodies.size());
                  reduced_arrays.reserve(array_bodies.size());
                  index_r2o_map.reserve(array_bodies.size());
                  for (size_t i = 0; i < array_bodies.size(); ++i) {
                      const ArrayBodyPtr& array_body = array_bodies[i];
                      if (array_body != nullptr) {
                          gsl::at(index_o2r_map, i) = index_r2o_map.size();
                          index_r2o_map.emplace_back(i);
                          reduced_arrays.emplace_back(array_body);
                      }
                  }
                  return std::make_tuple(std::move(reduced_arrays), std::move(index_r2o_map), std::move(index_o2r_map));
              };

              // Inputs
              std::vector<Array> reduced_input_arrays;
              std::vector<size_t> input_index_r2o_map;
              std::vector<absl::optional<size_t>> input_index_o2r_map;
              std::vector<ConstArrayRef> reduced_input_array_refs;
              std::tie(reduced_input_arrays, input_index_r2o_map, input_index_o2r_map) = get_reduced_arrays(inputs);
              CHAINERX_ASSERT(IsUniqueAndIncreasingIndexes(input_index_r2o_map, inputs.size()));
              CHAINERX_ASSERT(!reduced_input_arrays.empty());
              CHAINERX_ASSERT(std::all_of(reduced_input_arrays.begin(), reduced_input_arrays.end(), [](const Array& arr) {
                  return internal::GetArrayBody(arr) != nullptr;
              }));
              reduced_input_array_refs.insert(reduced_input_array_refs.begin(), reduced_input_arrays.begin(), reduced_input_arrays.end());

              // Outputs
              std::vector<Array> reduced_output_arrays;
              std::vector<size_t> output_index_r2o_map;
              std::vector<absl::optional<size_t>> output_index_o2r_map;
              std::vector<ConstArrayRef> reduced_output_array_refs;
              std::tie(reduced_output_arrays, output_index_r2o_map, output_index_o2r_map) = get_reduced_arrays(outputs);
              CHAINERX_ASSERT(IsUniqueAndIncreasingIndexes(output_index_r2o_map, outputs.size()));
              CHAINERX_ASSERT(!reduced_output_arrays.empty());
              CHAINERX_ASSERT(std::all_of(reduced_output_arrays.begin(), reduced_output_arrays.end(), [](const Array& arr) {
                  return internal::GetArrayBody(arr) != nullptr;
              }));
              reduced_output_array_refs.insert(
                      reduced_output_array_refs.begin(), reduced_output_arrays.begin(), reduced_output_arrays.end());

              // Insert backward function
              BackwardBuilder bb{"chainer_function", std::move(reduced_input_array_refs), std::move(reduced_output_array_refs)};
              if (BackwardBuilder::Target bt = bb.CreateTarget()) {
                  // Need to reallocate the function node in order to specify a custom deleter (that acquires the GIL before deletion).
                  auto function_node_ptr =
                          std::shared_ptr<py::object>{new py::object{std::move(function_node)}, [](gsl::owner<py::object*> ptr) {
                                                          py::gil_scoped_acquire acquire;
                                                          delete ptr;
                                                      }};

                  // Retain inputs/outputs
                  auto retain_arrays = [](auto retain,
                                          const std::vector<absl::optional<size_t>>& index_o2r_map,
                                          const std::vector<size_t>& indexes_to_retain) {
                      // Given the original indices to retain, retain the corresponding arrays and return the retain tokens. If the
                      // corresponding array was None, the token is nullopt.
                      using RetainedToken = decltype(retain(size_t{}));
                      std::vector<absl::optional<RetainedToken>> retained_tokens;
                      retained_tokens.reserve(indexes_to_retain.size());
                      for (size_t i : indexes_to_retain) {
                          if (auto j = index_o2r_map[i]) {
                              retained_tokens.emplace_back(retain(*j));
                          } else {
                              // Array to retain was None
                              retained_tokens.emplace_back(absl::nullopt);
                          }
                      }
                      return retained_tokens;
                  };

                  std::vector<absl::optional<RetainedInputToken>> retained_input_tokens =
                          retain_arrays([&bb](size_t j) { return bb.RetainInput(j); }, input_index_o2r_map, input_indexes_to_retain);
                  std::vector<absl::optional<RetainedOutputToken>> retained_output_tokens =
                          retain_arrays([&bb](size_t j) { return bb.RetainOutput(j); }, output_index_o2r_map, output_indexes_to_retain);

                  // Define backward function
                  bt.Define([chainer_output_count,
                             function_node_ptr = std::move(function_node_ptr),
                             input_index_r2o_map = std::move(input_index_r2o_map),
                             output_index_r2o_map = std::move(output_index_r2o_map),
                             in_toks = std::move(retained_input_tokens),
                             out_toks = std::move(retained_output_tokens)](BackwardContext& bctx) {
                      // Target input indices
                      // This is reduced <r> indices of grad-required inputs.
                      std::vector<size_t> target_input_indexes;
                      target_input_indexes.reserve(bctx.input_count());

                      for (size_t j = 0; j < bctx.input_count(); ++j) {
                          if (bctx.is_input_grad_required(j)) {
                              target_input_indexes.emplace_back(j);
                          }
                      }

                      CHAINERX_ASSERT(IsUniqueAndIncreasingIndexes(target_input_indexes, bctx.input_count()));

                      // Collect incoming output gradients
                      std::vector<ArrayBodyPtr> chainer_grad_outputs{chainer_output_count};
                      for (size_t j = 0; j < bctx.output_count(); ++j) {
                          if (auto& gy = bctx.output_grad(j)) {
                              size_t chainer_output_index = output_index_r2o_map[j];
                              ArrayBodyPtr gy_body = internal::GetArrayBody(*gy);
                              chainer_grad_outputs[chainer_output_index] = std::move(gy_body);
                          }
                      }

                      // Get retained inputs and outputs
                      auto get_retained_arrays = [](auto get_retained, const auto& tokens) {
                          std::vector<ArrayBodyPtr> retained_arrays;
                          retained_arrays.reserve(tokens.size());
                          for (const auto& tok : tokens) {
                              if (tok.has_value()) {
                                  // Retrieve the retained array.
                                  retained_arrays.emplace_back(internal::GetArrayBody(get_retained(*tok)));
                              } else {
                                  // Array to retain was None.
                                  retained_arrays.emplace_back(nullptr);
                              }
                          }
                          return retained_arrays;
                      };

                      std::vector<ArrayBodyPtr> retained_inputs =
                              get_retained_arrays([&bctx](const RetainedInputToken& tok) { return bctx.GetRetainedInput(tok); }, in_toks);
                      std::vector<ArrayBodyPtr> retained_outputs = get_retained_arrays(
                              [&bctx](const RetainedOutputToken& tok) { return bctx.GetRetainedOutput(tok); }, out_toks);

                      // Call FunctionNode._backward_chainerx()
                      std::vector<ArrayBodyPtr> chainer_grad_inputs;

                      {
                          // Target input indices that are passed to backward() of Chainer.
                          // This is indices of <grad-required> inputs among <all> of the inputs.
                          std::vector<size_t> chainer_target_input_indexes{};
                          chainer_target_input_indexes.reserve(target_input_indexes.size());
                          for (size_t j : target_input_indexes) {
                              chainer_target_input_indexes.emplace_back(input_index_r2o_map[j]);
                          }

                          py::gil_scoped_acquire acquire;
                          py::object func_backward = function_node_ptr->attr("_backward_chainerx");
                          py::object py_grad_inputs =
                                  func_backward(chainer_target_input_indexes, chainer_grad_outputs, retained_inputs, retained_outputs);
                          chainer_grad_inputs = py::cast<std::vector<ArrayBodyPtr>>(py_grad_inputs);
                      }
                      CHAINERX_ASSERT(chainer_grad_inputs.size() == target_input_indexes.size());

                      // Store computed input gradients
                      for (size_t k = 0; k < target_input_indexes.size(); ++k) {
                          size_t j = gsl::at(target_input_indexes, k);
                          ArrayBodyPtr& gx = gsl::at(chainer_grad_inputs, k);
                          CHAINERX_ASSERT(gx != nullptr);

                          bctx.input_grad(j) = Array{gx};
                      }
                  });
              }
              bb.Finalize();
          },
          "function_node"_a,
          "inputs"_a,
          "outputs"_a,
          "input_indexes_to_retain"_a,
          "output_indexes_to_retain"_a);
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
