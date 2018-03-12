#include "xchainer/python/slice.h"

#include <nonstd/optional.hpp>

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

Slice MakeSlice(const py::slice& slice) {
    auto py_slice_obj = reinterpret_cast<PySliceObject*>(slice.ptr());  // NOLINT: reinterpret_cast
    nonstd::optional<int64_t> start;
    nonstd::optional<int64_t> stop;
    nonstd::optional<int64_t> step;
    if (py_slice_obj->start != Py_None) {
        start.emplace(py::cast<int64_t>(py_slice_obj->start));
    }
    if (py_slice_obj->stop != Py_None) {
        stop.emplace(py::cast<int64_t>(py_slice_obj->stop));
    }
    if (py_slice_obj->step != Py_None) {
        step.emplace(py::cast<int64_t>(py_slice_obj->step));
    }
    return Slice(start, stop, step);
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
