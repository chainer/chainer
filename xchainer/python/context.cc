#include "xchainer/python/context.h"

#include <memory>
#include <sstream>

#include "xchainer/backend.h"
#include "xchainer/context.h"
#include "xchainer/device.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

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
    py::class_<Context>(m, "Context")
        .def(py::init())
        .def("get_backend", &Context::GetBackend, py::return_value_policy::reference)
        .def("get_device", [](Context& self, const std::string& device_name) -> Device& { return self.GetDevice(device_name); },
             py::return_value_policy::reference)
        .def("get_device",
             [](Context& self, const std::string& backend_name, int index) -> Device& {
                 return self.GetDevice({backend_name, index});
             },
             py::return_value_policy::reference);

    m.def("get_default_context", &GetDefaultContext, py::return_value_policy::reference);
    m.def("set_default_context", &SetDefaultContext);

    m.def("get_global_default_context", &GetGlobalDefaultContext, py::return_value_policy::reference);
    m.def("set_global_default_context", &SetGlobalDefaultContext);

    py::class_<PyContextScope>(m, "ContextScope").def("__enter__", &PyContextScope::Enter).def("__exit__", &PyContextScope::Exit);
    m.def("context_scope", [](Context& device) { return PyContextScope(device); });
}

}  // namespace xchainer
