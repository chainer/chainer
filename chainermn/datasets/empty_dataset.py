def create_empty_dataset(dataset):
    """Creates an empty dataset for models with no inputs and outputs.

    This function generates an empty dataset, i.e., ``__getitem__()`` only
    returns ``None``. Its dataset is compatible with the original one.
    Such datasets used for models which do not take any inputs,
    neither return any outputs. We expect models, e.g., whose ``forward()``
    is starting with ``chainermn.functions.recv()`` and ending with
    ``chainermn.functions.send()``.

    Args:
        dataset: Dataset to convert.

    Returns:
        ~chainer.datasets.TransformDataset:
            Dataset consists of only patterns in the original one.
    """
    return [()] * len(dataset)
