# Data-parallel seq2seq example (seq2seq.py)

A sample implementation of seq2seq model.

## Data download and setup

First, go to http://www.statmt.org/wmt15/translation-task.html#download and donwload necessary dataset.
Let's assume you are in a working directory called `$WMT_DIR`.

```bash
$ WMT_DIR=$HOME/path/to/your/data
$ cd $WMT_DIR
$ wget http://www.statmt.org/wmt10/training-giga-fren.tar
$ wget http://www.statmt.org/wmt15/dev-v2.tgz
$ tar -xf training-giga-fren.tar
$ tar -xf dev-v2.tgz
$ ls 
dev/  dev-v2.tgz  giga-fren.release2.fixed.en.gz  giga-fren.release2.fixed.fr.gz  training-giga-fren.tar

```

Next, you need to install required packages.

```bash
$ pip install nltk progressbar2
```

## Run

```bash
$ cd $CHAINERMN
$ mpiexec -n 2 python examples/seq2seq/seq2seq.py --gpu -c $WMT_DIR -i $WMT_DIR --out result.tmp
```

## Cache

Since the original dataset is huge, it is time-consuming to load the dataset at the beginning of the execution.
You can use cache to shorten the time after the second execution:

```bash
$ mpiexec -n 2 python examples/seq2seq/seq2seq.py --gpu -c $WMT_DIR -i $WMT_DIR --out result.tmp --cache $CACHE_DIR
```

`$CACHE_DIR` is an arbitrary directory where cache files will be generated.

# Model-parallel seq2seq example (seq2seq\_mp1.py)

First example for model-parallel seq2seq.
This example has two processes, one for encoder and the other for decoder.
To implement this kind of model-parallel RNN, you can use `create_model_parallel_n_step_rnn` as follows:

```python
class Encoder(chainer.Chain):

    def __init__(
            self, comm, n_layers, n_source_vocab, n_target_vocab, n_units):

        super(Encoder, self).__init__(
            embed_x=L.EmbedID(n_source_vocab, n_units),
            # Corresponding decoder LSTM will be invoked on process 1.
            mn_encoder=chainermn.links.create_multi_node_n_step_rnn(
                L.NStepLSTM(n_layers, n_units, n_units, 0.1),
                comm, rank_in=None, rank_out=1
            ),
        )
        self.comm = comm
        self.n_layers = n_layers
        self.n_units = n_units
        
 class Decoder(chainer.Chain):

    def __init__(
            self, comm, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Decoder, self).__init__(
            embed_y=L.EmbedID(n_target_vocab, n_units),
            # Corresponding encoder LSTM will be invoked on process 0.
            mn_decoder=chainermn.links.create_multi_node_n_step_rnn(
                L.NStepLSTM(n_layers, n_units, n_units, 0.1),
                comm, rank_in=0, rank_out=None),
            W=L.Linear(n_units, n_target_vocab),
        )
        self.comm = comm
        self.n_layers = n_layers
        self.n_units = n_units
```

Just wrapping the original RNN link by `create_multi_node_n_step_rnn`.

## Run

Note that this model can only invoked on two processes.

```bash
$ cd $CHAINERMN
$ mpiexec -n 2 python examples/seq2seq/seq2seq_mp1.py --gpu -c $WMT_DIR -i $WMT_DIR --out result.tmp
```
