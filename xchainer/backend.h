#pragma once

#include <memory>
#include <string>
#include <vector>

namespace xchainer {

class Device;

class Backend {
public:
    virtual ~Backend() = default;

    // Returns the number of available devices.
    //
    // This count is usually configurable by backend specific ways.
    virtual int GetDeviceCount() const = 0;

    // Returns the device for the given index.
    //
    // Throws out-of-index exception if index >= GetDeviceCount().
    virtual Device& GetDevice(int index) = 0;

    // Returns the name of this backend. This name should be unique within the context.
    virtual std::string GetName() const = 0;
};

}  // namespace xchainer
