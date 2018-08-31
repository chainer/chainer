#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace xchainer {
namespace error_detail {

inline void MakeMessageImpl(std::ostringstream& /*os*/) {}

// These two forward declarations are required to make the specializations visible from the generic version.
template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, int8_t first, const Args&... args);

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, uint8_t first, const Args&... args);

template <typename Arg, typename... Args>
void MakeMessageImpl(std::ostringstream& os, const Arg& first, const Args&... args) {
    os << first;
    MakeMessageImpl(os, args...);
}

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, int8_t first, const Args&... args) {
    os << static_cast<int>(first);
    MakeMessageImpl(os, args...);
}

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, uint8_t first, const Args&... args) {
    os << static_cast<unsigned int>(first);
    MakeMessageImpl(os, args...);
}

template <typename... Args>
std::string MakeMessage(const Args&... args) {
    std::ostringstream os;
    os << std::boolalpha;
    MakeMessageImpl(os, args...);
    return os.str();
}

}  // namespace error_detail

// All the exceptions defined in ChainerX must inherit this class.
class XchainerError : public std::runtime_error {
public:
    template <typename... Args>
    explicit XchainerError(const Args&... args) : runtime_error{error_detail::MakeMessage(args...)} {}
};

// Error on using invalid contexts.
class ContextError : public XchainerError {
public:
    using XchainerError::XchainerError;
};

// Error on using invalid backends.
class BackendError : public XchainerError {
public:
    using XchainerError::XchainerError;
};

// Error on using invalid devices.
class DeviceError : public XchainerError {
public:
    using XchainerError::XchainerError;
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

// Error on calling not-yet-implemented functionality
class NotImplementedError : public XchainerError {
public:
    using XchainerError::XchainerError;
};

// Error on failing gradient check
class GradientCheckError : public XchainerError {
public:
    using XchainerError::XchainerError;
};

}  // namespace xchainer
