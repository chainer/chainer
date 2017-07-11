Word2vec implementation with Chainer
*************************************

.. currentmodule:: chainer

0. Introduction
================

Word2vec is the tool for generating the distributed representation of words.
When the tool asssigns a real number vector to each word, the closer
the meanings of the words, the grater similarity the vectors will indicate.
As you know, "distributed representation" is assigning a real number vector for each
object and representing the object by the vector. When representatin a word
by distributed representation, we call it "distributed representation of words"
or "word embeddings". In this tutorial we will use the term "distributed
representation of words".

Let's think about what the meaning of words is. As you are a human, you can
understand that the words "animal" and "dog" are similar. But what information will
Word2vec use in order to learn the vectors of meanings? Such that the words
"animal" and "dog" are similar, but the words "food" and "dog" are not similar.

1. Basic idea
==============

Word2vec learns the similarity of word meanings from simple information. It learns
from a sequence of words in sentences. The idea is that the meaning of the word is
determined by the words around it. This idea is an old methodology, whith is called
"distributional hypothesis". It is mentioned in papers of 1950's[5]. The word to
be learned is called "Center Word", and the words around it are called
"Context Words". Depending on the window size ``c``, the number of Contex Words
will change.

For example, I will explain with the sentence "The cute cat jumps over the lazy dog.".

- All of the following figures consider "cat" as Center Word.
- According to the window size ``c``, you can see that Context Words are changing.

.. image:: ../../image/word2vec/center_context_word.png

2. Main algorithm
==================

- Word distribution expression creation tool 'word 2 vec' actually has two models built in Skip-gram and CBoW.
- I will explain with the figure below, but the meaning of the sign at that time is explained here.
    - N: Number of vocabulary.
    - D: Size of vector of distributed representation.
    - [tex: {v_t}]: Center Word. The size is [N, 1].
    - [tex: {v_ {t + c}}]: Context Word. The size is [N, 1].
    - [tex: {L_H}]: Input converted to distributed expression. The size is [D, 1].
    - [tex: {L_O}]: output layer. The size is [N, 1].
    - [tex: {W_H}]: Distributed representation matrix for input. The size is [N, D].
    - [tex: {W_O}]: Distributed representation matrix for the output. The size is [D, N].

2.1 Skip-gram
--------------

- Learn to predict Context Word [tex: {v_ {t + c}}] when Center Word [tex: {v_t}] is given
- At this time, each line of the distributed representation matrix W_H for Target Word becomes a distributed representation of each word.

2.2 Continuous Bag of Words (CBoW)
-----------------------------------

- Context Word [tex: {v_ {t + c}}] is given, learning to predict Center Word [tex: {v_t}]
- At this time, each column of the distributed representation matrix [tex: {W_O}] for the output becomes a distributed representation of each word.

2.3 Details of Skip-gram
-------------------------

- In this tutorial, Skip-gram is handled mainly from the following viewpoint.
    1. Learning algorithm is easier to understand than CBoW
    2. Even if the number of words increases, the accuracy is hard to fall down and it is easy to scale

2.4 Explanation using specific examples
----------------------------------------

- As in the example above, the number of vocabulary N is 10 and the size D of the vector of distributed representation is 2.
- Explain how to learn the word "dog" as Center Word and the word "animal" as Context Word.
- Since there should be more than one Context Word, repeat the following process for Context Word.
    1. The one - hot vector of the word dog is `[0 0 1 0 0 0 0 0 0 0]` and you enter it as Center Word.
    2. At this time, the third line of Distributed Expression Matrix [tex: {W_H}] for Center Word is the distributed representation of the word "dog" [tex: {L_H}].
        - By the way, the value here becomes a distributed expression of the word dog after learning.
    3. The output layer [tex: {L_O}] is the result of multiplying the distributed expression matrix [tex: {W_O}] for Context Word by the distributed expression of the word dog [tex: {L_H}].
    4. In order to limit the value of each element of the output layer, softmax function is applied to the output layer [tex: {L_O}] to calculate [tex: {W_O}]
       - Ultimately, it is necessary to update the parameter by making an error between the output layer and Context Word and back propagating the error back to the network.
       - However, the value of each element of the output layer takes a value from -∞ to + ∞ because the range is not limited. Context Word's one-hot vector takes only 0 or 1 for each element like `[1 0 0 0 0 0 0 0 0 0]`.
       - To limit the value of each element of the output layer to 0 to 1, apply softmax that limits the value of each element to the range 0 to 1.
    5. Calculate the error between [tex: {W_O}] and animal's one-hot vector` [1 0 0 0 0 0 0 0 0 0 0] `, and propagate the error back to the network to update the parameters

3. Implementation by Chainer
=============================

- There is a code related to word 2vec in examples on the GitHub repository, so we will explain based on that.
    - [https://github.com/chainer/chainer/tree/master/examples/word2vec:title]

3.1 Implementation method
--------------------------

- Basically if you use chainer you import in this way.
    
    - Importing functions like F, links like L makes it easy to use.
- Next is the definition of the network structure of skip-gram.
    -
    - We pass the vocabulary number `n_vocab`, the size of the dispersion vector` n_units`, and the loss function `loss_func` to the constructor` __init__`.
        - Parameter is being initialized in init_scope ().
            - It is recommended to initialize Parameter here.
            - Since we set Prameter as the attribute of Link, there are effects such as making IDE easier to follow code.
            - For details here [https://docs.chainer.org/en/stable/upgrade.html?highlight=init_scope#new-style-parameter-registration-apis-are-added-to-link:title]
        - Here the weight matrix `W` in` self.embed` is the distributed expression matrix [tex: {W_H}] for Center Word.
        - In case of Skip-gram, since we correspond one-to-one with Center Word and Context Word, there is no problem switching Context Word and Center Word, and changing Context Word and Center Word I am learning. (This is because it is easy to match the CBoW model and code.)
    - Function call `__call__` receives ID 'x` of Center Word, ID`context` of Context Word and returns error with loss function` loss_func`.
        - Obtaining the distributed representation corresponding to context with `e = self.embed (context)`.
        - Broad cast ID 'x' of Center Word for `batch_size` by the number of Context Word.
        - `x` is` [batch_size * n_context,] `,` e` is `[batch_size * n_context, n_units]`.
- Definition of the loss function. In effect, we are defining the network structure of skip-gram.
    -
    - After computing the linear mapping `self.out (x)` (`self.out: = L. Linear (n_in, n_out, initialW = 0) (x)`) with weight matrix for `x`, Calculate F.softmax_cross_entropy`.
    - Here, the linear mapping `self.out (x)` corresponds to the distributed expression matrix [tex: {W_O}] for Context Word, `F.softmax_cross_entropy` corresponds to the softmax function and loss calculation part.
- Definition of Iterator
    -
    - The constructor `__init__` is passed a document dataset` dataset` with a list of word ids, a window size `window`, and a mini batch size` batch_size`.
        - In this, we create a list `self.order` which shuffled the position of the word in the document. In order to learn, we do not learn from the beginning to the end of the document in order, but to select and learn words randomly from the document. The position of the word that cut off the beginning and the end by the window size is shuffled and entered.
        - Example: If the number of words in the document dataset `dataset` is 100 and the window size` window` is 5, `self.order` becomes numpy.array where numbers from 5 to 94 are shuffled.
    - Iterator definition `__next__` returns mini batch sized Center Word` center` and Context Word `context` according to the parameters of the constructor.
        - `position = self.order [i: i_end]` generates the index `position` of Center Word of` batch_size` minutes from the list `self.order` shuffling the position of the word. (`Position` will be converted to Center Word` center` later by `self.dataset.take`.)
        - `offset = np.concatenate ([np.arange (-w, 0), np.arange (1, w + 1)])` creates an offset `offset` representing the window.
        - `pos = position [:, None] + offset [None,:]` for each C

4.2 Run example
----------------

.. code-block:: console

    $ pwd
    /root2chainer/chainer/examples/word2vec
    $ $ python train_word2vec.py --test  # test modeで実行。全データで学習したいときは--testを消去
    GPU: -1
    # unit: 100
    Window: 5
    Minibatch-size: 1000
    # epoch: 20
    Training model: skipgram
    Output type: hsm
    
    n_vocab: 10000
    data length: 100
    epoch       main/loss   validation/main/loss
    1           4233.75     2495.33               
    2           1411.14     4990.66               
    3           4233.11     1247.66               
    4           2821.66     4990.65               
    5           4231.94     1247.66               
    6           5642.04     2495.3                
    7           5640.82     4990.64               
    8           5639.31     2495.28               
    9           2817.89     4990.62               
    10          1408.03     3742.94               
    11          5633.11     1247.62               
    12          4221.71     2495.21               
    13          4219.3      4990.56               
    14          4216.57     2495.16               
    15          4213.52     2495.12               
    16          5616.03     1247.55               
    17          5611.34     3742.78               
    18          2800.31     3742.74               
    19          1397.79     2494.95               
    20          2794.1      3742.66
