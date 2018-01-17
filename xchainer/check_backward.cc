#include "xchainer/check_backward.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_node.h"  // TODO(hvy): delete this line?
#include "xchainer/backprop.h"
#include "xchainer/error.h"
#include "xchainer/gradient_check.h"
#include "xchainer/numeric.h"

namespace xchainer {
namespace testing {

// TODO (hvy): Hide inside anonymous namespace

Arrays BackwardGradients(std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs) {
    Arrays outputs = func(inputs);

    const int nout = outputs.size();
    for (int i = 0; i < nout; ++i) {
        outputs[i].mutable_node()->set_grad(grad_outputs[i]);
    }

    // TODO(hvy): Only supporting functions with single outputs, support any number of outputs instead
    if (outputs.size() > 1) {
        throw NotImplementedError("Number of inputs must match the number of epsilon");
    }

    Backward(outputs[0]);
    Arrays backward_grads;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(backward_grads), [](const Array& input) { return *input.grad(); });
    return backward_grads;
}

void CheckBackwardComputation(std::function<Arrays(const Arrays&)> func, const Arrays& inputs, const Arrays& grad_outputs,
                              const Arrays& eps, float atol, float rtol) {
    const Arrays backward_grads = BackwardGradients(func, inputs, grad_outputs);
    const Arrays numerical_grads = CalculateNumericalGradient(func, inputs, grad_outputs, eps);
    assert(backward_grads.size() == numerical_grads.size());

    for (size_t i = 0; i < backward_grads.size(); ++i) {
        if (!AllClose(backward_grads[i], numerical_grads[i], atol, rtol)) {
            // TODO(hyv): Use EXCEPT_* or FAIL and print proper outputs
            throw GradientCheckError("too large error");
        }
    }
}

}  // namespace testing
}  // namespace xchainer
