import io
import sys
import threading

import six

from chainer.dataset import dataset_mixin


class TextDataset(dataset_mixin.DatasetMixin):

    """Dataset of a line-oriented text file.

    This dataset reads each line of text file(s) on every call of the
    :meth:`__getitem__` operator.
    Positions of line boundaries are cached so that you can quickliy
    random access the text file by the line number.

    .. note::
        Cache will be built in the constructor.
        You can pickle and unpickle the dataset to reuse the cache, but in
        that case you are responsible to guarantee that files are not
        modified after the cache has built.

    Args:
        paths (str or list of str):
            Path to the text file(s).
            If it is a string, this dataset reads a line from the text file
            and emits it as :class:`str`.
            If it is a list of string, this dataset reads lines from each
            text file and emits it as a tuple of :class:`str`. In this case,
            number of lines in all files must be the same.
        encoding (str or list of str):
            Name of the encoding used to decode the file.
            See the description in :func:`open` for the supported options and
            how it works.
            When reading from multiple text files, you can also pass a list of
            :class:`str` to use different encoding for each file.
        errors (str or list of str):
            String that specifies how decoding errors are to be handled.
            See the description in :func:`open` for the supported options and
            how it works.
            When reading from multiple text files, you can also pass a list of
            :class:`str` to use different error handling policy for each file.
        newline (str or list of str):
            Controls how universal newlines mode works.
            See the description in :func:`open` for the supported options and
            how it works.
            When reading from multiple text files, you can also pass a list of
            :class:`str` to use different mode for each file.
        filter_func (callable):
            Function to filter each line of the text file.
            It should be a function that takes number of arguments equals to
            the number of files. Arguments are lines loaded from each file.
            The filter function must return True to accept the line, or
            return False to skip the line.

    """

    def __init__(
            self, paths, encoding=None, errors=None, newline=None,
            filter_func=None):
        if isinstance(paths, six.string_types):
            paths = [paths]
        elif not paths:
            raise ValueError('at least one text file must be specified')

        if isinstance(encoding, six.string_types) or encoding is None:
            encoding = [encoding] * len(paths)
        if isinstance(errors, six.string_types) or errors is None:
            errors = [errors] * len(paths)
        if isinstance(newline, six.string_types) or newline is None:
            newline = [newline] * len(paths)

        if not (len(paths) == len(encoding) == len(errors) == len(newline)):
            raise ValueError(
                'length of each option must match with the number of '
                'text files to read')

        self._paths = paths
        self._encoding = encoding
        self._errors = errors
        self._newline = newline
        self._fps = None

        self._open()

        # Line number is 0-origin.
        # `lines` is a list of line numbers not filtered; if no filter_func is
        # given, it is range(linenum)).
        # `bounds` is a list of cursor positions of line boundaries for each
        # file, i.e. i-th line of k-th file starts at `bounds[k][i]`.
        linenum = 0
        lines = []
        bounds = tuple([[0] for _ in self._fps])
        while True:
            data = [fp.readline() for fp in self._fps]
            if not all(data):  # any of files reached EOF
                if any(data):  # not all files reached EOF
                    raise ValueError(
                        'number of lines in files does not match')
                break
            for i, fp in enumerate(self._fps):
                bounds[i].append(fp.tell())
            if filter_func is not None and filter_func(*data):
                lines.append(linenum)
            linenum += 1

        if filter_func is None:
            lines = six.moves.range(linenum)

        self._bounds = bounds
        self._lines = lines
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_fps']
        del state['_lock']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._open()
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._lines)

    def _open(self):
        self._fps = [
            io.open(
                path,
                mode='rt',
                encoding=encoding,
                errors=errors,
                newline=newline,
            ) for path, encoding, errors, newline in
            six.moves.zip(self._paths, self._encoding, self._errors,
                          self._newline)
        ]

    def close(self):
        """Manually closes all text files.

        In most cases, you do not have to call this method, because files will
        automatically be closed after TextDataset instance goes out of scope.
        """
        exc = None
        for fp in self._fps:
            try:
                fp.close()
            except Exception:
                exc = sys.exc_info()
        if exc is not None:
            six.reraise(*exc)

    def get_example(self, idx):
        if idx < 0 or len(self._lines) <= idx:
            raise IndexError
        linenum = self._lines[idx]

        self._lock.acquire()
        try:
            for k, fp in enumerate(self._fps):
                fp.seek(self._bounds[k][linenum])
            lines = [fp.readline() for fp in self._fps]
            if len(lines) == 1:
                return lines[0]
            return tuple(lines)
        finally:
            self._lock.release()
