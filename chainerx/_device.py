import chainerx


def _recover_device(backend_name, device_index):
    # Recovers the device instance.
    # This function is used together with chainerx.Device.__reduce__.
    # TODO(niboshi): Save the context name and lookup the context with it.
    context = chainerx.get_default_context()
    backend = context.get_backend(backend_name)
    device = backend.get_device(device_index)
    return device
