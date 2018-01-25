import collections


Query = collections.namedtuple('Query', ['sentence', 'answer', 'fact'])
Sentence = collections.namedtuple('Sentence', ['sentence'])




def split(sentence):
    return sentence.lower().replace('.', '').replace('?', '').split()


def convert(vocab, words):
    return [vocab[w] for w in words]


def parse_line(vocab, line):
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


def parse_data(vocab, lines):
    data = []
    all_data = []
    for line in lines:
        sid, content = line.strip().split(' ', 1)
        if sid == '1':
            if len(data) > 0:
                all_data.append(data)
                data = []

        data.append(parse_line(vocab, content))

    if len(data) > 0:
        all_data.append(data)

    return all_data


def read_data(vocab, path):
    with open(path) as f:
        return parse_data(vocab, f)
