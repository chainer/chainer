#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace xchainer {

class Device;

// Backend base class.
class Backend {
public:
    virtual ~Backend();

    // Returns the name of this backend. This name should be unique within the context.
    virtual std::string GetName() const = 0;

    // Returns the number of available devices.
    //
    // This count is usually configurable by backend specific ways.
    virtual int GetDeviceCount() const = 0;

    // Returns the device for the given index.
    //
    // Throws out_of_range exception if index >= GetDeviceCount().
    Device& GetDevice(int index);

private:
    // Creates a new device.
    // This function is called from GetDevice().
    virtual std::unique_ptr<Device> CreateDevice(int index) = 0;

    std::vector<std::unique_ptr<Device>> devices_;

    std::mutex devices_mutex_;
};

}  // namespace xchainer
