#include <pybind11/pybind11.h>

PYBIND11_MODULE(_core, m) {
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer._core.*"
}
