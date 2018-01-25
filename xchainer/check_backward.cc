#include "xchainer/check_backward.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/backprop.h"
#include "xchainer/error.h"
#include "xchainer/gradient_check.h"
#include "xchainer/numeric.h"

namespace xchainer {
namespace {

std::vector<nonstd::optional<Array>> BackwardGradients(std::function<std::vector<Array>(const std::vector<Array>&)> func,
                                                       const std::vector<Array>& inputs, const std::vector<Array>& grad_outputs,
                                                       const std::string& graph_name) {
    std::vector<Array> outputs = func(inputs);

    const int nout = outputs.size();
    for (int i = 0; i < nout; ++i) {
        outputs[i].MutableNode(graph_name)->set_grad(grad_outputs[i]);
    }

    // TODO(hvy): Currently only supporting functions with single outputs, support any number of outputs instead
    if (outputs.size() > 1) {
        throw NotImplementedError("Functions with more than one output are not supported");
    }
    Backward(outputs[0], graph_name);

    std::vector<nonstd::optional<Array>> backward_grads;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(backward_grads), [&graph_name](const Array& input) {
        return input.RequiresGrad(graph_name) ? nonstd::optional<Array>(input.Grad(graph_name)) : nonstd::nullopt;
    });

    return backward_grads;
}

}  // namespace

void CheckBackwardComputation(std::function<std::vector<Array>(const std::vector<Array>&)> func, const std::vector<Array>& inputs,
                              const std::vector<Array>& grad_outputs, const std::vector<Array>& eps, double atol, double rtol,
                              const std::string& graph_name) {
    const std::vector<Array> numerical_grads = CalculateNumericalGradient(func, inputs, grad_outputs, eps);
    const std::vector<nonstd::optional<Array>> backward_grads = BackwardGradients(func, inputs, grad_outputs, graph_name);
    ASSERT_EQ(backward_grads.size(), numerical_grads.size());

    std::ostringstream failure_os;
    const int nin = backward_grads.size();
    for (int i = 0; i < nin; ++i) {
        if (backward_grads[i]) {  // All inputs do not necessarily require gradients
            if (!AllClose(*backward_grads[i], numerical_grads[i], atol, rtol)) {
                failure_os << "Backward check failure on input " << i << " (Total inputs: " << inputs.size() << ")\n"
                           << "Graph name: " << graph_name << "\n"
                           << "Atol: " << atol << "\n"
                           << "Rtol: " << rtol << "\n"
                           << "Eps (perturbation):\n"
                           << eps[i] << "\n"
                           << "Backward gradients:\n"
                           << *backward_grads[i] << "\n"
                           << "Numerical gradients:\n"
                           << numerical_grads[i];
            }
        }
    }

    // Do nothing if all backward-numerical gradient pairs were close, else generate a nonfatal failure
    std::string failure_message = failure_os.str();
    if (!failure_message.empty()) {
        ADD_FAILURE() << failure_message;
    }
}

}  // namespace xchainer
