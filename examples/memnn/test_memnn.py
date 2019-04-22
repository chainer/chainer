#!/usr/bin/env python

import argparse

import numpy

import chainer

import babi
import memnn


def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: End-to-end memory networks')
    parser.add_argument('MODEL',
                        help='Path to model directory specified with `-m` '
                        'argument in the training script')
    parser.add_argument('DATA',
                        help='Path to test data in bAbI dataset '
                        '(e.g. "qa1_single-supporting-fact_test.txt")')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)
    xp = device.xp
    device.use()

    model, vocab = memnn.load_model(args.MODEL)
    model.to_device(device)

    network = model.predictor
    max_memory = network.max_memory
    id_to_vocab = {i: v for v, i in vocab.items()}

    test_data = babi.read_data(vocab, args.DATA)
    print('Test data: %s: %d' % (args.DATA, len(test_data)))

    sentence_len = max(max(len(s.sentence) for s in story)
                       for story in test_data)
    correct = total = 0
    for story in test_data:
        mem = xp.zeros((max_memory, sentence_len), dtype=numpy.int32)
        i = 0
        for sent in story:
            if isinstance(sent, babi.Sentence):
                if i == max_memory:
                    mem[0:i - 1, :] = mem[1:i, :]
                    i -= 1
                mem[i, 0:len(sent.sentence)] = xp.asarray(sent.sentence)
                i += 1
            elif isinstance(sent, babi.Query):
                query = xp.array(sent.sentence, dtype=numpy.int32)

                # networks assumes mini-batch data
                score = network(mem[None], query[None])[0]
                answer = int(xp.argmax(score.array))

                if answer == sent.answer:
                    correct += 1
                total += 1
                print(id_to_vocab[answer], id_to_vocab[sent.answer])

    accuracy = float(correct) / total
    print('Accuracy: %.2f%%' % (accuracy * 100))


if __name__ == '__main__':
    main()
