#!/usr/bin/env python
import argparse
import json
import sys

import chainer

import nets
import nlp_utils


def setup_model(args):
    sys.stderr.write(json.dumps(args.__dict__, indent=2) + '\n')
    setup = json.load(open(args.model_setup))
    sys.stderr.write(json.dumps(setup, indent=2) + '\n')

    vocab = json.load(open(setup['vocab_path']))
    n_class = setup['n_class']

    # Setup a model
    if setup['model'] == 'rnn':
        Encoder = nets.RNNEncoder
    elif setup['model'] == 'cnn':
        Encoder = nets.CNNEncoder
    elif setup['model'] == 'bow':
        Encoder = nets.BOWMLPEncoder
    encoder = Encoder(n_layers=setup['layer'], n_vocab=len(vocab),
                      n_units=setup['unit'], dropout=setup['dropout'])
    model = nets.TextClassifier(encoder, n_class)
    chainer.serializers.load_npz(setup['model_path'], model)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    return model, vocab, setup


def run_online(gpu):
    # predict labels online
    for l in sys.stdin:
        l = l.strip()
        if not l:
            print('# blank line')
            continue
        text = nlp_utils.normalize_text(l)
        words = nlp_utils.split_text(text, char_based=setup['char_based'])
        xs = nlp_utils.transform_to_array([words], vocab, with_label=False)
        xs = nlp_utils.convert_seq(xs, device=gpu, with_label=False)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            prob = model.predict(xs, softmax=True)[0]
        answer = int(model.xp.argmax(prob))
        score = float(prob[answer])
        print('{}\t{:.4f}\t{}'.format(answer, score, ' '.join(words)))


def run_batch(gpu, batchsize=64):
    # predict labels by batch

    def predict_batch(words_batch):
        xs = nlp_utils.transform_to_array(words_batch, vocab, with_label=False)
        xs = nlp_utils.convert_seq(xs, device=gpu, with_label=False)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            probs = model.predict(xs, softmax=True)
        answers = model.xp.argmax(probs, axis=1)
        scores = probs[model.xp.arange(answers.size), answers].tolist()
        for words, answer, score in zip(words_batch, answers, scores):
            print('{}\t{:.4f}\t{}'.format(answer, score, ' '.join(words)))

    batch = []
    for l in sys.stdin:
        l = l.strip()
        if not l:
            if batch:
                predict_batch(batch)
                batch = []
            print('# blank line')
            continue
        text = nlp_utils.normalize_text(l)
        words = nlp_utils.split_text(text, char_based=setup['char_based'])
        batch.append(words)
        if len(batch) >= batchsize:
            predict_batch(batch)
            batch = []
    if batch:
        predict_batch(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model-setup', required=True,
                        help='Model setup dictionary.')
    args = parser.parse_args()

    model, vocab, setup = setup_model(args)
    if args.gpu >= 0:
        run_batch(args.gpu)
    else:
        run_online(args.gpu)
