#include "xchainer/python/context.h"

#include <memory>
#include <sstream>
#include <string>

#include "xchainer/backend.h"
#include "xchainer/context.h"
#include "xchainer/device.h"

#include "xchainer/python/common.h"
#include "xchainer/python/device.h"

namespace xchainer {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

Context& GetContext(py::handle handle) {
    if (handle.is_none()) {
        return GetDefaultContext();
    }

    if (py::isinstance<Context&>(handle)) {
        return py::cast<Context&>(handle);
    }

    throw py::type_error{"Invalid Context type: " + py::cast<std::string>(py::repr(handle))};
}

class PyContextScope {
public:
    explicit PyContextScope(Context& target) : target_(target) {}
    void Enter() { scope_ = std::make_unique<ContextScope>(target_); }
    void Exit(py::args args) {
        (void)args;  // unused
        scope_.reset();
    }

private:
    // TODO(beam2d): better to replace it by "optional"...
    std::unique_ptr<ContextScope> scope_;
    Context& target_;
};

void InitXchainerContext(pybind11::module& m) {
    py::class_<Context> c{m, "Context"};
    c.def(py::init());
    c.def("get_backend", &Context::GetBackend, py::return_value_policy::reference);
    c.def("get_device",
          [](Context& self, const std::string& device_name) -> Device& { return self.GetDevice(device_name); },
          py::return_value_policy::reference);
    c.def("get_device",
          [](Context& self, const std::string& backend_name, int index) -> Device& {
              return self.GetDevice({backend_name, index});
          },
          py::return_value_policy::reference);

    m.def("get_backend", &GetBackend, py::return_value_policy::reference);
    m.def("get_device",
          [](py::handle device) -> Device& { return GetDevice(device); },
          py::arg("device") = nullptr,
          py::return_value_policy::reference);
    m.def("get_device",
          [](const std::string& backend_name, int index) -> Device& {
              return xchainer::GetDevice({backend_name, index});
          },
          py::return_value_policy::reference);

    m.def("get_default_context", &GetDefaultContext, py::return_value_policy::reference);
    m.def("set_default_context", &SetDefaultContext);

    m.def("get_global_default_context", &GetGlobalDefaultContext, py::return_value_policy::reference);
    m.def("set_global_default_context", &SetGlobalDefaultContext);
}

void InitXchainerContextScope(pybind11::module& m) {
    py::class_<PyContextScope> c(m, "ContextScope");
    c.def("__enter__", &PyContextScope::Enter);
    c.def("__exit__", &PyContextScope::Exit);

    m.def("context_scope", [](Context& context) { return PyContextScope(context); });
}

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
