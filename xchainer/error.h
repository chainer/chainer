#pragma once

#include <stdexcept>

namespace xchainer {

// All the exceptions defined in xChainer must inherit this class.
class XchainerError : public std::runtime_error {
public:
    using runtime_error::runtime_error;
};

// Error on shape mismatch, invalid strides, and so on.
class DimensionError : public XchainerError {
public:
    using XchainerError::XchainerError;
};

// Error on dtype mismatch.
class DtypeError : public XchainerError {
public:
    using XchainerError::XchainerError;
};

}  // namespace xchainer
