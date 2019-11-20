#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace chainerx {
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
class ChainerxError : public std::runtime_error {
public:
    template <typename... Args>
    explicit ChainerxError(const Args&... args) : runtime_error{error_detail::MakeMessage(args...)} {}
};

// Error on using invalid contexts.
class ContextError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;
};

// Error on using invalid backends.
class BackendError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;
};

// Error on using invalid devices.
class DeviceError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;
};

// Error on out of range and too many indices for array.
class IndexError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;
};

// Error on shape mismatch, invalid strides, and so on.
class DimensionError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;
};

// Error on dtype mismatch.
class DtypeError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;
};

// Error on calling not-yet-implemented functionality
class NotImplementedError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;
};

// Error on mismatch gradient traits
class GradientError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;
};

// Error on failing gradient check
class GradientCheckError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;
};

}  // namespace chainerx
