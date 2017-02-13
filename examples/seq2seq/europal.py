import collections
import re
import gzip

import numpy
import progressbar


split_pattern = re.compile('([.,!?"\':;)(])')
digit_pattern = re.compile(r'\d')


def split_sentence(s):
    s = s.lower()
    s = split_pattern.sub(r' \1', s)
    s = digit_pattern.sub('0', s)
    words = s.strip().split()
    return [w for w in words if w]


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
    token_count = 0
    unknown_count = 0
    for words in read_file(path):
        array = make_array(word_id, words)
        dataset.append(array)
        token_count += array.size
        unknown_count += (array == 1).sum()
    print('# of tokens: %d' % token_count)
    print('# of unknown: %d (%.2f %%)'
          % (unknown_count, 100. * unknown_count / token_count))
    return dataset


def make_array(word_id, words):
    ids = [word_id.get(word, 1) for word in words]
    return numpy.array(ids, 'i')


if __name__ == '__main__':
    vocab = count_words('wmt/giga-fren.release2.fixed.en')
    make_dataset('wmt/giga-fren.release2.fixed.en', vocab)
