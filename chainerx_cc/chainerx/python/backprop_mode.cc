#include "chainerx/python/common_export.h"

#include "chainerx/python/backprop_mode.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/backprop_mode.h"
#include "chainerx/context.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"

#include "chainerx/python/common.h"
#include "chainerx/python/context.h"

namespace chainerx {
namespace python {
namespace python_internal {
namespace {

namespace py = pybind11;  // standard convention
using py::literals::operator""_a;

template <class BackpropModeScope>
class PyBackpropModeScope {
public:
    explicit PyBackpropModeScope(std::vector<BackpropId> backprop_ids) : backprop_ids_{std::move(backprop_ids)} {}
    explicit PyBackpropModeScope(Context& context) : context_{&context} {}

    void Enter() {
        if (backprop_ids_.has_value()) {
            scope_ = std::make_unique<BackpropModeScope>(*backprop_ids_);
        } else {
            CHAINERX_ASSERT(context_ != nullptr);
            scope_ = std::make_unique<BackpropModeScope>(*context_);
        }
    }
    void Exit(py::args args) {
        (void)args;  // unused
        scope_.reset();
    }

private:
    // optional requires having copy ctor, so use unique_ptr instead
    std::unique_ptr<BackpropModeScope> scope_;
    Context* context_{nullptr};
    absl::optional<std::vector<BackpropId>> backprop_ids_{};
};

template <class BackpropModeScope>
void InitChainerxBackpropModeScope(pybind11::module& m, const char* class_name, const char* function_name) {
    py::class_<PyBackpropModeScope<BackpropModeScope>> c{m, class_name};
    c.def("__enter__", &PyBackpropModeScope<BackpropModeScope>::Enter);
    c.def("__exit__", &PyBackpropModeScope<BackpropModeScope>::Exit);

    m.def(function_name, [](const BackpropId& backprop_id) { return PyBackpropModeScope<BackpropModeScope>{{backprop_id}}; });
    m.def(function_name, [](const std::vector<BackpropId>& backprop_ids) { return PyBackpropModeScope<BackpropModeScope>{backprop_ids}; });
    m.def(function_name,
          [](py::object context) { return PyBackpropModeScope<BackpropModeScope>{GetContext(context)}; },
          "context"_a = py::none());
}

}  // namespace

void InitChainerxBackpropMode(pybind11::module& m) {
    InitChainerxBackpropModeScope<NoBackpropModeScope>(m, "NoBackpropMode", "no_backprop_mode");
    InitChainerxBackpropModeScope<ForceBackpropModeScope>(m, "ForceBackpropMode", "force_backprop_mode");

    m.def("is_backprop_required", [](const BackpropId& backprop_id) { return IsBackpropRequired(backprop_id); }, "backprop_id"_a);
    m.def("is_backprop_required", [](py::handle context) { return IsBackpropRequired(GetContext(context)); }, "context"_a = py::none());
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
