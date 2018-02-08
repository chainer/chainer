# End-to-end memory networks example

This is an example of end-to-end memory networks[1].


## Preparation

Run `download.py` script to download bAbI dataset.

```
$ ./download.py
```

You can find `tasks_1-20_v1-2.tar.gz` in the current directory. Unpack it.

```
$ tar zxf tasks_1-20_v1-2.tar.gz
```

It uncompresses files into `tasks_1-20_v1-2` directory. It contains 20 subtasks.


## Train

Run `train_memnn.py` script with training data and test data.

```
$ ./train_memnn.py tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt
```

Add `--gpu` argument to specify GPU ID if you want to use GPU.
Add `--model` argument to specify a directory where you want to store trained model. It will be used in test section.

```
$ ./train_memnn.py tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt \
  --gpu 0 --model model1
```


## Test

You can run `test_memnn.py` script to evaluate trained model stored with `--model` argument.

```
$ ./test_memnn.py model1 tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt
```

You can append `--gpu` argument to use GPU.



[1] Sainbayar Sukhbaatar, Arthur szlam, Jason Weston and Rob Fergus. End-To-End Memory Networks. https://papers.nips.cc/paper/5846-end-to-end-memory-networks
