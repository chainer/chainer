import collections


Query = collections.namedtuple('Query', ['sentence', 'answer', 'fact'])
Sentence = collections.namedtuple('Sentence', ['sentence'])


def split(sentence):
    """Splits a sentence into words.

    Args:
        sentence (str): A sentence to split.

    Returns:
        list of str: A list of words. Punctuations are removed.

    """
    return sentence.lower().replace('.', '').replace('?', '').split()


def convert(vocab, words):
    """Converts a word list into a word ID list.

    Args:
        vocab (collections.defaultdict): A dictionary storing word IDs.
        words (list of str): A list of wards to convert.

    Returns:
        list of int: A list of word IDs.

    """
    return [vocab[w] for w in words]


def parse_line(vocab, line):
    """Parses each line and make a named tuple.

    Args:
        vocab (collections.defaultdict): A dictionary storing word IDs.
        line (str): A line to parse in bAbI dataset.

    Returns:
        Query or Sentence: Parsed tuple.

    """
    if '\t' in line:
        # question line
        question, answer, fact_id = line.split('\t')
        aid = convert(vocab, [answer])[0]
        words = split(question)
        wid = convert(vocab, words)
        ids = list(map(int, fact_id.split(' ')))
        return Query(wid, aid, ids)

    else:
        # sentence line
        words = split(line)
        wid = convert(vocab, words)
        return Sentence(wid)


def read_data(vocab, path):
    """Reads a bAbI dataset.

    Args:
        vocab (collections.defaultdict): A dictionary storing word IDs.
        path (str): Path to bAbI data file.

    Returns:
        list of Query of Sentence: Parsed lines.

    """
    data = []
    all_data = []
    with open(path) as f:
        for line in f:
            sid, content = line.strip().split(' ', 1)
            if sid == '1':
                if data:
                    all_data.append(data)
                    data = []

            data.append(parse_line(vocab, content))

        if data:
            all_data.append(data)

        return all_data
