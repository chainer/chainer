import csv
import glob
import io
import os
import tarfile

import numpy
from six.moves.urllib import request

from nlp_utils import make_vocab
from nlp_utils import normalize_text
from nlp_utils import split_text
from nlp_utils import transform_to_array


def download_dbpedia():
    request.urlretrieve(
        'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz',  # NOQA
        'dbpedia_csv.tar.gz')
    tf = tarfile.open('dbpedia_csv.tar.gz', 'r')
    tf.extractall('./')
    os.remove('dbpedia_csv.tar.gz')


def read_dbpedia(path, shrink=1, char_based=False):
    dataset = []
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for i, (label, title, text) in enumerate(csv.reader(f)):
            if i % shrink != 0:
                continue
            label = int(label) - 1  # Index begins from 1
            tokens = split_text(normalize_text(text), char_based)
            dataset.append((tokens, label))
    return dataset


def get_dbpedia(path='./dbpedia_csv', vocab=None, shrink=1, char_based=False):
    if not os.path.isdir(path):
        print('download dbpedia')
        download_dbpedia()
    train_path = os.path.join(path, 'train.csv')
    test_path = os.path.join(path, 'test.csv')

    print('read dbpedia')
    train = read_dbpedia(train_path, shrink=shrink, char_based=char_based)
    test = read_dbpedia(test_path, shrink=shrink, char_based=char_based)

    if vocab is None:
        print('constract vocabulary based on frequency')
        vocab = make_vocab(train)

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab


def download_imdb():
    request.urlretrieve(
        'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
        'aclImdb_v1.tar.gz')
    tf = tarfile.open('aclImdb_v1.tar.gz', 'r')
    tf.extractall('./')
    os.remove('aclImdb_v1.tar.gz')


def read_imdb(base_path, shrink=1, fine_grained=False, char_based=False):
    fg_label_dict = {'1': 0, '2': 0, '3': 1, '4': 1,
                     '7': 2, '8': 2, '9': 3, '10': 3}

    def read_and_label(path, label):
        dataset = []
        for i, f_path in enumerate(glob.glob(os.path.join(path, '*'))):
            if i % shrink != 0:
                continue
            with io.open(f_path, encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            tokens = split_text(normalize_text(text), char_based)
            if fine_grained:
                # extract from f_path. e.g. /pos/200_8.txt -> 8
                label = fg_label_dict[f_path.split('_')[-1][:-4]]
                dataset.append((tokens, label))
            else:
                dataset.append((tokens, label))
        return dataset

    pos_dataset = read_and_label(os.path.join(base_path, 'pos'), 0)
    neg_dataset = read_and_label(os.path.join(base_path, 'neg'), 1)
    return pos_dataset + neg_dataset


def get_imdb(path='./aclImdb', vocab=None, shrink=1, fine_grained=False,
             char_based=False):
    if not os.path.isdir(path):
        print('download imdb')
        download_imdb()
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')

    print('read imdb')
    train = read_imdb(train_path, shrink=shrink, fine_grained=fine_grained,
                      char_based=char_based)
    test = read_imdb(test_path, shrink=shrink, fine_grained=fine_grained,
                     char_based=char_based)

    if vocab is None:
        print('constract vocabulary based on frequency')
        vocab = make_vocab(train)

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab


def download_other_dataset(path, name):
    os.mkdir(path)
    base_url = 'https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/'  # NOQA
    if name in ['custrev', 'mpqa', 'rt-polarity', 'subj']:
        files = [name + '.all']
    elif name == 'TREC':
        files = [name + suff for suff in ['.train.all', '.test.all']]
    else:
        files = [name + suff for suff in ['.train', '.test']]
    for f_name in files:
        request.urlretrieve(
            os.path.join(base_url, f_name),
            os.path.join(path, f_name))


def read_other_dataset(path, shrink=1, char_based=False):
    dataset = []
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for i, l in enumerate(f):
            if i % shrink != 0 or not len(l.strip()) >= 3:
                continue
            label, text = l.strip().split(None, 1)
            label = int(label)
            tokens = split_text(normalize_text(text), char_based)
            dataset.append((tokens, label))
    return dataset


def get_other_text_dataset(name, path=None, vocab=None, shrink=1,
                           char_based=False, seed=777):
    assert(name in ['TREC', 'stsa.binary', 'stsa.fine',
                    'custrev', 'mpqa', 'rt-polarity', 'subj'])

    if path is None:
        path = './' + name

    if not os.path.isdir(path):
        print('download ' + name)
        download_other_dataset(path, name)

    print('read ' + name)
    if name in ['custrev', 'mpqa', 'rt-polarity', 'subj']:
        print('split {} to train:test=9:1 by rand seed {}'.format(name, seed))
        numpy.random.seed(seed)
        alldata = numpy.random.permutation(
            read_other_dataset(os.path.join(path, name + '.all'),
                               shrink=shrink, char_based=char_based))
        train = alldata[:-len(alldata) // 10]
        test = alldata[-len(alldata) // 10:]
    elif name == 'TREC':
        train = read_other_dataset(os.path.join(path, name + '.train.all'),
                                   shrink=shrink, char_based=char_based)
        test = read_other_dataset(os.path.join(path, name + '.test.all'),
                                  shrink=shrink, char_based=char_based)
    else:
        train = read_other_dataset(os.path.join(path, name + '.train'),
                                   shrink=shrink, char_based=char_based)
        test = read_other_dataset(os.path.join(path, name + '.test'),
                                  shrink=shrink, char_based=char_based)

    if vocab is None:
        print('constract vocabulary based on frequency')
        vocab = make_vocab(train)

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab
