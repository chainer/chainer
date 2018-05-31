#pragma once

#include <vector>

#include "xchainer/array.h"
#include "xchainer/constant.h"

namespace xchainer {

enum class DoubleBackpropOption : bool {
    kDisable = false,
    kEnable = true,
};

class BackardContext;

namespace backward_detail {

using ArrayRef = std::reference_wrapper<Array>;
using ConstArrayRef = std::reference_wrapper<const Array>;

using BackwardFunc = std::function<void(BackwardContext&)>;

}  // namespace backward_detail

class BackwardContext {
public:
    // Returns whether the output has a propagated gradient.
    bool HasOutputGrad(int output_index) const;

    // Return the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const Array& GetOutputGrad(int output_index) const;

    // Stores the computed input gradient.
    void SetInputGrad(int input_index, int output_index, const Array& grad_input);

private:
    // Holds zero-filled arrays for outputs without actual gradientss.
    // The arrays are allocated on-demand in GetOutputGrad.
    mutable std::vector<nonstd::optional<Array>> zero_output_arrays_;
};

class DefineBackwardScope {
public:
    DefineBackwardScope(const char* func_name, std::initializer_list<backward_detail::ConstArrayRef> outputs);

    // Defines a backward function with respect to specified input arrays.
    // It throws XchainerError if a backward function has already been defined for any of specified input arrays.
    void Define(std::initializer_list<backward_detail::ConstArrayRef> inputs, backward_detail::BackwardFunc&& backward_func);

private:
    const char* func_name_;
    std::vector<backward_detail::ConstArrayRef> output_arrays_;
    std::unordered_set<backward_detail::ConstArrayRef> defined_input_arrays_;
};

void Backward(
        const Array& output,
        const GraphId& graph_id = kDefaultGraphId,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

void Backward(
        const std::vector<ConstArrayRef>& outputs,
        const GraphId& graph_id = kDefaultGraphId,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

}  // namespace xchainer
