import json
import os

import numpy
import six
from six.moves.urllib import request

from chainer import dataset


class PTBWordsBase(dataset.Dataset):

    def __init__(self):
        self._data, self.n_vocab = _load(self._type)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i, ...]


class PTBWordsTraining(PTBWordsBase):

    """Penn Tree Bank training dataset as one long word sequence.

    TODO(bema2d): document it.

    """
    name = 'PTBWords_train'
    _type = 'train'


class PTBWordsValidation(PTBWordsBase):

    """Penn Tree Bank validation dataset as one long word sequence.

    TODO(beam2d): document it.

    """
    name = 'PTBWords_valid'
    _type = 'valid'


class PTBWordsTest(PTBWordsBase):

    """Penn Tree Bank test dataset as one long word sequence.

    TODO(beam2d): document it.

    """
    name = 'PTBWords_test'
    _type = 'test'


_urltmpl = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/{}'
_root = dataset.get_dataset_path('ptb_words')
_vocab_path = os.path.join(_root, 'vocab.json')


def _load(name):
    try:
        npz = numpy.load(_get_npz_path(name))
        data = npz['x']
        n_vocab = npz['n_vocab']
    except IOError:
        d, n_vocab = _download_all()
        data = d[name]

    return data, n_vocab


def _download_all():
    vocab = {}
    ret = {name: _load_data(name, vocab) for name in ('train', 'valid', 'test')}
    for name, data in six.iteritems(ret):
        numpy.savez_compressed(_get_npz_path(name), x=data, n_vocab=len(vocab))

    with open(os.path.join(_root, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)

    return ret, len(vocab)


def _load_data(name, vocab):
    words = _download(name)
    data = numpy.ndarray(len(words), dtype=numpy.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        data[i] = vocab[word]
    return data


def _download(name):
    print('Downloading PTB {} dataset...'.format(name))
    try:
        os.makedirs(_root)
    except:
        pass
    filename = 'ptb.{}.txt'.format(name)
    txtpath = os.path.join(_root, filename)
    request.urlretrieve(_urltmpl.format(filename), txtpath)

    with open(txtpath) as txt:
        return txt.read().replace('\n', '<eos>').strip().split()


def _read_words(words, vocab):
    data = numpy.ndarray(len(words), dtype=numpy.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        data[i] = vocab[word]
    return data


def _get_npz_path(name):
    return os.path.join(_root, name + '.npz')
