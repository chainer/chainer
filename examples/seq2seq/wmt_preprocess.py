#!/usr/bin/env python

from __future__ import unicode_literals

import argparse
import collections
import io
import re

import progressbar


split_pattern = re.compile(r'([.,!?"\':;)(])')
digit_pattern = re.compile(r'\d')


def split_sentence(s, use_lower):
    if use_lower:
        s = s.lower()
    s = s.replace('\u2019', "'")
    s = digit_pattern.sub('0', s)
    words = []
    for word in s.strip().split():
        words.extend(split_pattern.split(word))
    words = [w for w in words if w]
    return words


def count_lines(path):
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        return sum([1 for _ in f])


def read_file(path, use_lower):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for line in bar(f, max_value=n_lines):
            words = split_sentence(line, use_lower)
            yield words


def proc_dataset(
        path, outpath, vocab_path=None, vocab_size=None, use_lower=False):
    token_count = 0
    counts = collections.Counter()
    with io.open(outpath, 'w', encoding='utf-8') as f:
        for words in read_file(path, use_lower):
            line = ' '.join(words)
            f.write(line)
            f.write('\n')
            if vocab_path:
                for word in words:
                    counts[word] += 1
            token_count += len(words)
    print('number of tokens: %d' % token_count)

    if vocab_path and vocab_size:
        vocab = [word for (word, _) in counts.most_common(vocab_size)]
        with io.open(vocab_path, 'w', encoding='utf-8') as f:
            for word in vocab:
                f.write(word)
                f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'INPUT', help='input sentence data')
    parser.add_argument(
        'OUTPUT', help='output sentence data')
    parser.add_argument(
        '--vocab-file', help='vocabulary file to save')
    parser.add_argument(
        '--vocab-size', type=int, default=40000,
        help='size of vocabulary file')
    parser.add_argument(
        '--lower', action='store_true', help='use lower case')
    args = parser.parse_args()

    proc_dataset(
        args.INPUT, args.OUTPUT, vocab_path=args.vocab_file,
        vocab_size=args.vocab_size, use_lower=args.lower)


if __name__ == '__main__':
    main()
