import collections
import gzip

import numpy
import progressbar


def split_sentence(s):
    return s.split()


def open_file(path):
    if path.endswith('.gz'):
        return gzip.open(path)
    else:
        return open(path)


def count_lines(path):
    with open_file(path) as f:
        return sum([1 for _ in f])


def read_file(path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar(max_value=n_lines)
    with open_file(path) as f:
        for line in bar(f):
            words = split_sentence(line)
            yield words


def count_words(path):
    counts = collections.Counter()
    for words in read_file(path):
        for word in words:
            counts[word] += 1

    vocab = [word for (word, _) in counts.most_common(40000)]
    return vocab


def make_dataset(path, vocab):
    word_id = {word: index for index, word in enumerate(vocab)}
    dataset = []
    for words in read_file(path):
        array = make_array(word_id, words)
        dataset.append(array)
    return dataset


def make_array(word_id, words):
    ids = [word_id[word] if word in word_id else 0 for word in words]
    return numpy.array(ids, 'i')


if __name__ == '__main__':
    vocab = count_words('wmt/giga-fren.release2.fixed.en')
    make_dataset('wmt/giga-fren.release2.fixed.en', vocab)
