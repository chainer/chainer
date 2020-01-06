#include "chainerx/python/common_export.h"

#include "chainerx/python/graph.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "chainerx/backprop_scope.h"
#include "chainerx/context.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"

#include "chainerx/python/common.h"
#include "chainerx/python/context.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention
using py::literals::operator""_a;

class PyBackpropScope {
public:
    explicit PyBackpropScope(std::string backprop_name, Context& context) : backprop_name_{std::move(backprop_name)}, context_{context} {}
    BackpropId Enter() {
        if (scope_ != nullptr) {
            throw ChainerxError{"Backprop scope cannot be nested."};
        }
        if (exited_) {
            throw ChainerxError{"Exited backprop scope cannot be reused."};
        }
        scope_ = std::make_unique<BackpropScope>(backprop_name_, context_);
        return scope_->backprop_id();
    }
    void Exit(py::args args) {
        (void)args;  // unused
        exited_ = true;
        scope_.reset();
    }

private:
    std::string backprop_name_;
    Context& context_;
    std::unique_ptr<BackpropScope> scope_;
    bool exited_{false};
};

void InitChainerxGraph(pybind11::module& m) {
    py::class_<AnyGraph>{m, "AnyGraph"};  // NOLINT(misc-unused-raii,bugprone-unused-raii)

    // TODO(imanishi): Add module function to retrieve default backprop id.
    m.attr("anygraph") = AnyGraph{};

    py::class_<BackpropId> c{m, "BackpropId"};
    c.def("__repr__", [](const BackpropId& backprop_id) {
        std::ostringstream stream;
        stream << backprop_id;
        return stream.str();
    });
    c.def_property_readonly("context", &BackpropId::context);
    c.def_property_readonly("name", &BackpropId::GetName);
}

void InitChainerxBackpropScope(pybind11::module& m) {
    py::class_<PyBackpropScope> c{m, "BackpropScope"};
    c.def("__enter__", &PyBackpropScope::Enter);
    c.def("__exit__", &PyBackpropScope::Exit);

    m.def("backprop_scope",
          [](const std::string& backprop_name, py::handle context) { return PyBackpropScope(backprop_name, GetContext(context)); },
          "backprop_name"_a,
          "context"_a = py::none());
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
