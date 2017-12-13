#pragma once

#include <stdexcept>

namespace xchainer {

// All the exceptions defined in Xchainer must inherit this class.
class XchainerError : public std::runtime_error {
public:
    using runtime_error::runtime_error;
};

// Error on dtype mismatch.
class DtypeError : public XchainerError {
public:
    using XchainerError::XchainerError;
};

}  // namespace xchainer
