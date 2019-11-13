// Every source file in this directory should include this file before anything else.

#pragma once

#include <pybind11/pybind11.h>

namespace chainerx {

class PYBIND11_EXPORT ChainerxError;
class PYBIND11_EXPORT ContextError;
class PYBIND11_EXPORT BackendError;
class PYBIND11_EXPORT DeviceError;
class PYBIND11_EXPORT IndexError;
class PYBIND11_EXPORT DimensionError;
class PYBIND11_EXPORT DtypeError;
class PYBIND11_EXPORT NotImplementedError;
class PYBIND11_EXPORT GradientError;
class PYBIND11_EXPORT GradientCheckError;

}  // namespace chainerx
