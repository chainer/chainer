

class BaseIxIndexer(object):
    """Base class for IxIndexer
    """
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, item):
        raise NotImplementedError
