# Sequence-to-sequnce learning example for machine translation

This is a minimal example of sequence-to-sequence learning. Sequence-to-sequence learning model is a conversion model from a sequence to a sequence. You can regard many tasks in natural language processing field as this type of task, such as machine translation, dialogue and summarization.

This example sciprt is simplest one. An input sequence is processed by a stacked LSTM-RNN and it is encoded as a fixed-size of vector. And the an output sequence is also processed by another stacked LSTM-RNN. In decoding time, an output sequence is generated in argmax fashion.


## Dataset format

You need to prepare four files.

1. Source language sentence file
2. Source language vocabulary file
3. Targe language sentence file
4. Target language vocabulary file

In sentence files each line represents a sentece. In each line, each sentence needs to be separated into words by space characters.
Each line corresponds to other one. So both file need to have the same numbers of lines.

In vocabulary files each line represents a word. Words which are not in the vocabulary files are treated as special words `<UNKNOWN>`.


## Tarin with WMT dataset

First you need to prepare parallel corpus. Download 10^9 French-English corpus from WMT15 website.

http://www.statmt.org/wmt15/translation-task.html

```
$ http://www.statmt.org/wmt10/training-giga-fren.tar
$ tar -xf training-giga-fren.tar
$ ls
giga-fren.release2.fixed.en.gz  giga-fren.release2.fixed.fr.gz
$ gunzip giga-fren.release2.fixed.en.gz
$ gunzip giga-fren.release2.fixed.fr.gz
```

Then run the preprocess script `wmt_preprocess.py` to make sentence files and vocabulary files.

```
$ python wmt_preprocess.py giga-fren.release2.fixed.en giga-fren.preprocess.en \
  --vocab-file vocab.en
$ python wmt_preprocess.py giga-fren.release2.fixed.en giga-fren.preprocess.fr \
  --vocab-file vocab.fr
```

Now you can get four files:

- Source sentence file: `giga-fren.preprocess.en`
- Source vocabulary file: `vocab.en`
- Target sentence file: `giga-fren.preprocess.fr`
- Source vocabulary file: `vocab.fr`

Of course you can apply arbitrary preprocess by making a script.

For validation, get news article 2013 data and run preprocessor for them:

```
$ wget http://www.statmt.org/wmt15/dev-v2.tgz
$ tar zxf dev-v2.tgz
$ python wmt_preprocess.py dev/newstest2013.en newstest2013.preprocess.en
$ python wmt_preprocess.py dev/newstest2013.fr newstest2013.preprocess.fr
```

Let's start training. Add `--validation-source` and `--validation-target` argument
to specify validation dataset.

```
$ python seq2seq.py --gpu=0 giga-fen.preprocess.en giga-fren.preprocess.fr \
vocab.en vocab.fr \
--validation-source newstest2013.preprocess.en \
--validation-target newstest2013.preprocess.fr
```

See command line help for other options.
