class Query(object):

    def __init__(self, sentence, answer, fact):
        self.sentence = sentence
        self.answer = answer
        self.fact = fact


class Sentence(object):

    def __init__(self, sentence):
        self.sentence = sentence


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
    last_id = 0
    for line in lines:
        line = line.strip()
        pos = line.find(' ')
        sid = int(line[:pos])
        if sid == 1:
            if len(data) > 0:
                all_data.append(data)
                data = []
                last_id = 0

        assert sid == last_id + 1
        data.append(parse_line(vocab, line[pos + 1:]))
        last_id = sid

    if len(data) > 0:
        all_data.append(data)

    return all_data


def read_data(vocab, path):
    with open(path) as f:
        return parse_data(vocab, f)
