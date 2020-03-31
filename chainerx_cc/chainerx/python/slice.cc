#include "chainerx/python/common_export.h"

#include "chainerx/python/slice.h"

#include <absl/types/optional.h>

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

Slice MakeSlice(const py::slice& slice) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto* py_slice = reinterpret_cast<const PySliceObject*>(slice.ptr());
    auto to_optional = [](PyObject* var) -> absl::optional<int64_t> {
        if (var == Py_None) {
            return absl::nullopt;
        }
        return py::cast<int64_t>(var);
    };
    return Slice{to_optional(py_slice->start), to_optional(py_slice->stop), to_optional(py_slice->step)};
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
