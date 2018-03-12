#include "xchainer/python/slice.h"

#include <nonstd/optional.hpp>

#include "xchainer/slice.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;

void InitXchainerSlice(pybind11::module& m) {
    py::class_<Slice>{m, "Slice"}
            .def(py::init([](const nonstd::optional<int64_t>& stop) { return Slice(nonstd::nullopt, stop, nonstd::nullopt); }))
            .def(py::init([](const nonstd::optional<int64_t>& start, const nonstd::optional<int64_t>& stop) {
                return Slice(start, stop, nonstd::nullopt);
            }))
            .def(py::init([](const nonstd::optional<int64_t>& start,
                             const nonstd::optional<int64_t>& stop,
                             const nonstd::optional<int64_t>& step) { return Slice(start, stop, step); }))
            .def(py::init([](const py::slice& slice) {
                auto py_slice_obj = reinterpret_cast<PySliceObject*>(slice.ptr());
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
            }))
            .def_property_readonly("start", &Slice::start)
            .def_property_readonly("stop", &Slice::stop)
            .def_property_readonly("step", &Slice::step);
}

}  // namespace xchainer
