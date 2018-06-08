# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import pickle
import unittest

import six

from chainer import datasets
from chainer import testing


class TestTextDataset(unittest.TestCase):

    def setUp(self):
        self.root = os.path.join(os.path.dirname(__file__), 'text_dataset')

    def _dataset(self, path, **kwargs):
        def _absolute(p):
            return '{}{}{}'.format(self.root, os.sep, p)

        if isinstance(path, six.string_types):
            path = _absolute(path)
        else:
            path = [_absolute(p) for p in path]

        return datasets.TextDataset(path, **kwargs)

    def test_close(self):
        ds = self._dataset('ascii_1.txt')
        assert ds[0] == 'hello\n'
        ds.close()
        with self.assertRaises(ValueError):
            ds[0]

    def test_close_exception(self):
        ds = self._dataset(['ascii_1.txt', 'ascii_1.txt', 'ascii_1.txt'])
        assert not ds._fps[0].closed
        assert not ds._fps[1].closed
        assert not ds._fps[2].closed
        ds._fps[1] = None
        with self.assertRaises(AttributeError):
            ds.close()
        assert ds._fps[0].closed
        assert ds._fps[2].closed

    def test_len(self):
        ds = self._dataset('ascii_1.txt')
        assert len(ds) == 3

    def test_len_noeol(self):
        # No linefeed at the end of the file.
        ds = self._dataset('ascii_noeol.txt', encoding=['ascii'])
        assert len(ds) == 3

    def test_len_unicode(self):
        ds = self._dataset(['utf8_1.txt'], encoding='utf-8')
        assert len(ds) == 3

    def test_len_multiple(self):
        ds = self._dataset(['utf8_1.txt', 'utf8_2.txt'], encoding='utf-8')
        assert len(ds) == 3

    def test_get(self):
        ds = self._dataset(['ascii_1.txt'])
        assert ds[0] == 'hello\n'
        assert ds[1] == 'world\n'
        assert ds[2] == 'test\n'

    def test_get_unicode(self):
        ds = self._dataset(['utf8_1.txt'], encoding='utf-8')
        assert ds[0] == 'テスト1\n'
        assert ds[1] == 'テスト2\n'
        assert ds[2] == 'Test3\n'

    def test_get_crlf(self):
        ds = self._dataset(['utf8_crlf.txt'], encoding='utf-8')
        assert ds[0] == 'テスト1\n'
        assert ds[1] == 'テスト2\n'
        assert ds[2] == 'Test3\n'

    def test_get_multiple(self):
        ds = self._dataset(['utf8_1.txt', 'utf8_2.txt'], encoding='utf-8')
        assert ds[0] == ('テスト1\n', 'Test1\n')
        assert ds[1] == ('テスト2\n', 'テスト2\n')
        assert ds[2] == ('Test3\n', 'テスト3\n')

    def test_get_blank(self):
        # File with blank (empty) line.
        ds = self._dataset(['ascii_blank_line.txt'], encoding='ascii')
        assert ds[0] == 'hello\n'
        assert ds[1] == 'world\n'
        assert ds[2] == '\n'
        assert ds[3] == 'test\n'

    def test_encoding(self):
        # UTF-8 with BOM
        ds = self._dataset(['utf8sig.txt'], encoding='utf-8-sig')
        assert ds[0] == 'テスト1\n'
        assert ds[1] == 'Test2\n'
        assert ds[2] == 'Test3\n'

    def test_encoding_multiple(self):
        ds = self._dataset(
            ['ascii_1.txt', 'utf8_1.txt'],
            encoding=['ascii', 'utf-8'])
        assert ds[0] == ('hello\n', 'テスト1\n')
        assert ds[1] == ('world\n', 'テスト2\n')
        assert ds[2] == ('test\n', 'Test3\n')

    def test_errors(self):
        ds = self._dataset(
            ['utf8_1.txt'], encoding='ascii', errors='ignore')
        assert ds[0] == '1\n'  # "テスト" is ignored
        assert ds[1] == '2\n'
        assert ds[2] == 'Test3\n'

    def test_newline(self):
        # CRLF
        ds = self._dataset(['utf8_crlf.txt'], encoding='utf-8', newline='\r\n')
        assert ds[0] == 'テスト1\r\n'
        assert ds[1] == 'テスト2\r\n'
        assert ds[2] == 'Test3\r\n'

    def test_filter(self):
        def _filter(line):
            return line != 'world\n'
        ds = self._dataset(['ascii_1.txt'], filter_func=_filter)
        assert len(ds) == 2
        assert ds[0] == 'hello\n'
        assert ds[1] == 'test\n'

    def test_filter_multiple(self):
        def _filter(s1, s2):
            return s1 != 'world\n' and 'test' in s2
        ds = self._dataset(['ascii_1.txt', 'ascii_2.txt'], filter_func=_filter)
        assert len(ds) == 2
        assert ds[0] == ('hello\n', 'test file\n')
        assert ds[1] == ('test\n', 'world test\n')

    def test_pickle_unpickle(self):
        ds1 = self._dataset(['utf8_1.txt', 'utf8_2.txt'], encoding='utf-8')
        assert ds1[0] == ('テスト1\n', 'Test1\n')
        ds2 = pickle.loads(pickle.dumps(ds1))
        assert ds1[1] == ('テスト2\n', 'テスト2\n')
        assert ds2[1] == ('テスト2\n', 'テスト2\n')


testing.run_module(__name__, __file__)
