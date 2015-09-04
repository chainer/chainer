class Serializer(object):

    @property
    def reader(self):
        return not self.writer

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        raise NotImplementedError
