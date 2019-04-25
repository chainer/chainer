def _format_array_props(arrays):
    # Formats array shapes and dtypes for error messages.
    assert isinstance(arrays, (list, tuple))

    return ', '.join([
        None if arr is None
        else '{}:{}'.format(arr.shape, arr.dtype.name)
        for arr in arrays])
