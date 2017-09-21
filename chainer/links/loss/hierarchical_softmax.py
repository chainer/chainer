import copy

import numpy
import six

from chainer import cuda
from chainer import function
from chainer.initializers import uniform
from chainer import link
from chainer.utils import type_check
from chainer import variable


LEAF = -1
NOT_LEAF = -1
FINISH_SAMPLING = -2


class TreeParser(object):

    def __init__(self):
        self.next_id = 0

    def size(self):
        return self.next_id

    def get_paths(self):
        return self.paths

    def get_codes(self):
        return self.codes

    def get_parent2child(self):
        return self.parent2child

    def get_node2word(self):
        return self.node2word

    def parse(self, tree):
        self.next_id = 0
        self.path = []
        self.code = []
        self.paths = {}
        self.codes = {}
        self.parent2child = {}
        self.node2word = {}
        self._parse(tree)

        assert(len(self.path) == 0)
        assert(len(self.code) == 0)
        assert(len(self.paths) == len(self.codes))

    def _parse(self, node):
        if isinstance(node, tuple):
            # internal node
            if len(node) != 2:
                raise ValueError(
                    'All internal nodes must have two child nodes')
            left, right = node
            node_id = self.next_id
            self.path.append(node_id)

            self.next_id += 1
            self.code.append(1.0)
            left_id, left_word = self._parse(left)

            self.code[-1] = -1.0
            right_id, right_word = self._parse(right)

            self.node2word[node_id] = (left_word, right_word)
            self.parent2child[node_id] = (left_id, right_id)

            self.path.pop()
            self.code.pop()
            return node_id, NOT_LEAF

        else:
            # leaf node
            self.paths[node] = numpy.array(self.path, dtype=numpy.int32)
            self.codes[node] = numpy.array(self.code, dtype=numpy.float32)
            return LEAF, node


class BinaryHierarchicalSoftmaxFunction(function.Function):

    """Hierarchical softmax function based on a binary tree.

    This function object should be allocated beforehand, and be copied on every
    forward computation, since the initializer parses the given tree. See the
    implementation of :class:`BinaryHierarchicalSoftmax` for details.

    Args:
        tree: A binary tree made with tuples like ``((1, 2), 3)``.

    .. seealso::
       See :class:`BinaryHierarchicalSoftmax` for details.

    """

    def __init__(self, tree):
        parser = TreeParser()
        parser.parse(tree)
        paths = parser.get_paths()
        codes = parser.get_codes()
        parent2child = parser.get_parent2child()
        node2word = parser.get_node2word()
        n_vocab = max(paths.keys()) + 1
        n_node = len(node2word)
        self.n_vocab = n_vocab
        self.paths = numpy.concatenate(
            [paths[i] for i in six.moves.range(n_vocab) if i in paths])
        self.codes = numpy.concatenate(
            [codes[i] for i in six.moves.range(n_vocab) if i in codes])

        begins = numpy.empty((n_vocab + 1,), dtype=numpy.int32)
        begins[0] = 0
        for i in six.moves.range(0, n_vocab):
            length = len(paths[i]) if i in paths else 0
            begins[i + 1] = begins[i] + length
        self.begins = begins

        self.parent2child = numpy.array(
            [parent2child[i] for i in six.moves.range(n_node)], 'i')
        self.node2word = numpy.array(
            [node2word[i] for i in six.moves.range(n_node)], 'i')

        self.parser_size = parser.size()

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, t_type, w_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 2,
            t_type.dtype == numpy.int32,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
            w_type.dtype == numpy.float32,
            w_type.ndim == 2,
            w_type.shape[0] == self.parser_size,
            w_type.shape[1] == x_type.shape[1],
        )

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            self.paths = cuda.to_gpu(self.paths)
            self.codes = cuda.to_gpu(self.codes)
            self.begins = cuda.to_gpu(self.begins)
            self.parent2child = cuda.to_gpu(self.parent2child)
            self.node2word = cuda.to_gpu(self.node2word)

    def to_cpu(self):
        self.paths = cuda.to_cpu(self.paths)
        self.codes = cuda.to_cpu(self.codes)
        self.begins = cuda.to_cpu(self.begins)
        self.parent2child = cuda.to_cpu(self.parent2child)
        self.node2word = cuda.to_cpu(self.node2word)

    def forward_cpu(self, inputs):
        x, t, W = inputs

        loss = numpy.float32(0.0)
        for ix, it in six.moves.zip(x, t):
            loss += self._forward_cpu_one(ix, it, W)
        return numpy.array(loss),

    def _forward_cpu_one(self, x, t, W):
        begin = self.begins[t]
        end = self.begins[t + 1]

        w = W[self.paths[begin:end]]
        wxy = w.dot(x) * self.codes[begin:end]
        loss = numpy.logaddexp(0.0, -wxy)  # == log(1 + exp(-wxy))
        return numpy.sum(loss)

    def backward_cpu(self, inputs, grad_outputs):
        x, t, W = inputs
        gloss, = grad_outputs
        gx = numpy.empty_like(x)
        gW = numpy.zeros_like(W)
        for i, (ix, it) in enumerate(six.moves.zip(x, t)):
            gx[i] = self._backward_cpu_one(ix, it, W, gloss, gW)
        return gx, None, gW

    def _backward_cpu_one(self, x, t, W, gloss, gW):
        begin = self.begins[t]
        end = self.begins[t + 1]

        path = self.paths[begin:end]
        w = W[path]
        wxy = w.dot(x) * self.codes[begin:end]
        g = -gloss * self.codes[begin:end] / (1.0 + numpy.exp(wxy))
        gx = g.dot(w)
        gw = g.reshape((g.shape[0], 1)).dot(x.reshape(1, len(x)))
        gW[path] += gw
        return gx

    def forward_gpu(self, inputs):
        x, t, W = inputs
        max_length = cuda.reduce(
            'T t, raw T begins', 'T out', 'begins[t + 1] - begins[t]',
            'max(a, b)', 'out = a', '0',
            'binary_hierarchical_softmax_max_length')(t, self.begins)
        max_length = cuda.to_cpu(max_length)[()]

        length = max_length * x.shape[0]
        ls = cuda.cupy.empty((length,), dtype=numpy.float32)
        n_in = x.shape[1]
        wxy = cuda.cupy.empty_like(ls)
        cuda.elementwise(
            '''raw T x, raw T w, raw int32 ts, raw int32 paths,
            raw T codes, raw int32 begins, int32 c, int32 max_length''',
            'T ls, T wxy',
            '''
            int ind = i / max_length;
            int offset = i - ind * max_length;
            int t = ts[ind];

            int begin = begins[t];
            int length = begins[t + 1] - begins[t];

            if (offset < length) {
              int p = begin + offset;
              int node = paths[p];

              T wx = 0;
              for (int j = 0; j < c; ++j) {
                int w_ind[] = {node, j};
                int x_ind[] = {ind, j};
                wx += w[w_ind] * x[x_ind];
              }
              wxy = wx * codes[p];
              ls = log(1 + exp(-wxy));
            } else {
              ls = 0;
            }
            ''',
            'binary_hierarchical_softmax_forward'
        )(x, W, t, self.paths, self.codes, self.begins, n_in, max_length, ls,
          wxy)
        self.max_length = max_length
        self.wxy = wxy
        return ls.sum(),

    def backward_gpu(self, inputs, grad_outputs):
        x, t, W = inputs
        gloss, = grad_outputs

        n_in = x.shape[1]
        gx = cuda.cupy.zeros_like(x)
        gW = cuda.cupy.zeros_like(W)
        cuda.elementwise(
            '''T wxy, raw T x, raw T w, raw int32 ts, raw int32 paths,
            raw T codes, raw int32 begins, raw T gloss,
            int32 c, int32 max_length''',
            'raw T gx, raw T gw',
            '''
            int ind = i / max_length;
            int offset = i - ind * max_length;
            int t = ts[ind];

            int begin = begins[t];
            int length = begins[t + 1] - begins[t];

            if (offset < length) {
              int p = begin + offset;
              int node = paths[p];
              T code = codes[p];

              T g = -gloss[0] * code / (1.0 + exp(wxy));
              for (int j = 0; j < c; ++j) {
                int w_ind[] = {node, j};
                int x_ind[] = {ind, j};
                atomicAdd(&gx[x_ind], g * w[w_ind]);
                atomicAdd(&gw[w_ind], g * x[x_ind]);
              }
            }
            ''',
            'binary_hierarchical_softmax_bwd'
        )(self.wxy, x, W, t, self.paths, self.codes, self.begins, gloss, n_in,
          self.max_length, gx, gW)
        return gx, None, gW


class BinaryHierarchicalSoftmax(link.Link):

    """Hierarchical softmax layer over binary tree.

    In natural language applications, vocabulary size is too large to use
    softmax loss.
    Instead, the hierarchical softmax uses product of sigmoid functions.
    It costs only :math:`O(\\log(n))` time where :math:`n` is the vocabulary
    size in average.

    At first a user need to prepare a binary tree whose each leaf is
    corresponding to a word in a vocabulary.
    When a word :math:`x` is given, exactly one path from the root of the tree
    to the leaf of the word exists.
    Let :math:`\\mbox{path}(x) = ((e_1, b_1), \\dots, (e_m, b_m))` be the path
    of :math:`x`, where :math:`e_i` is an index of :math:`i`-th internal node,
    and :math:`b_i \\in \\{-1, 1\\}` indicates direction to move at
    :math:`i`-th internal node (-1 is left, and 1 is right).
    Then, the probability of :math:`x` is given as below:

    .. math::

       P(x) &= \\prod_{(e_i, b_i) \\in \\mbox{path}(x)}P(b_i | e_i)  \\\\
            &= \\prod_{(e_i, b_i) \\in \\mbox{path}(x)}\\sigma(b_i x^\\top
               w_{e_i}),

    where :math:`\\sigma(\\cdot)` is a sigmoid function, and :math:`w` is a
    weight matrix.

    This function costs :math:`O(\\log(n))` time as an average length of paths
    is :math:`O(\\log(n))`, and :math:`O(n)` memory as the number of internal
    nodes equals :math:`n - 1`.

    Args:
        in_size (int): Dimension of input vectors.
        tree: A binary tree made with tuples like `((1, 2), 3)`.

    Attributes:
        W (~chainer.Variable): Weight parameter matrix.

    See: Hierarchical Probabilistic Neural Network Language Model [Morin+,
    AISTAT2005].

    .. admonition:: Example

        Let an a dictionary containing word counts ``word_counts`` be:

        >>> word_counts = {0: 8, 1: 5, 2: 6, 3: 4, 4: 10, 5: 1, 6: 32, 7: 21}
        >>> tree = BinaryHierarchicalSoftmax.create_huffman_tree(word_counts)
        >>> hsm = BinaryHierarchicalSoftmax(3, tree)

    """

    def __init__(self, in_size, tree):
        # This function object is copied on every forward computation.
        super(BinaryHierarchicalSoftmax, self).__init__()
        self._func = BinaryHierarchicalSoftmaxFunction(tree)
        self.tree = tree
        with self.init_scope():
            self.W = variable.Parameter(uniform.Uniform(1),
                                        (self._func.parser_size, in_size))

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            super(BinaryHierarchicalSoftmax, self).to_gpu(device)
            self._func.to_gpu(device)

    def to_cpu(self):
        super(BinaryHierarchicalSoftmax, self).to_cpu()
        self._func.to_cpu()

    @staticmethod
    def create_huffman_tree(word_counts):
        """Makes a Huffman tree from a dictionary containing word counts.

        This method creates a binary Huffman tree, that is required for
        :class:`BinaryHierarchicalSoftmax`.
        For example, ``{0: 8, 1: 5, 2: 6, 3: 4}`` is converted to
        ``((3, 1), (2, 0))``.

        Args:
            word_counts (dict of int key and int or float values):
                Dictionary representing counts of words.

        Returns:
            Binary Huffman tree with tuples and keys of ``word_coutns``.

        """
        if len(word_counts) == 0:
            raise ValueError('Empty vocabulary')

        q = six.moves.queue.PriorityQueue()
        # Add unique id to each entry so that we can compare two entries with
        # same counts.
        # Note that itreitems randomly order the entries.
        for uid, (w, c) in enumerate(six.iteritems(word_counts)):
            q.put((c, uid, w))

        while q.qsize() >= 2:
            (count1, id1, word1) = q.get()
            (count2, id2, word2) = q.get()
            count = count1 + count2
            tree = (word1, word2)
            q.put((count, min(id1, id2), tree))

        return q.get()[2]

    def sample(self, x):
        """Sample an example for a given input from the tree.

        Args:
            x (~chainer.Variable): Input variable for sample word ids.

        Returns:
            array: Array of word indexes in a binary tree ``self.tree``.

        .. admonition:: Example

            Let an input vector ``x`` be:

            >>> word_cnts = {0: 8, 1: 5, 2: 6, 3: 4, 4: 10, 5: 1, 6: 32}
            >>> tree = BinaryHierarchicalSoftmax.create_huffman_tree(word_cnts)
            >>> hsm = BinaryHierarchicalSoftmax(3, tree)
            >>> x = np.array([[0.2, 0.2, 0.3], [0.1, 0.3, 0.1]], dtype='f')
            >>> hsm.sample(Variable(x))
            [0, 3]


        """
        if len(self.tree) == 0:
            raise ValueError('Empty tree')

        xp = cuda.get_array_module(x)
        parent2child = self._func.parent2child
        node2word = self._func.node2word
        batchsize = len(x)
        start_ids = xp.zeros(batchsize, 'i')
        list_next_ids = []
        list_sampled_ids = []

        def _sigmoid(x):
            half = x.dtype.type(0.5)
            return self.xp.tanh(x * half) * half + half

        rows = xp.arange(batchsize, dtype=xp.int32)
        while True:
            w = self.W.data[start_ids]
            x_t = xp.transpose(x.data)
            score = xp.sum(xp.dot(w, x_t), axis=1)
            prob_left = _sigmoid(score)[:, None]
            prob_right = 1 - prob_left
            prob = xp.concatenate([prob_left, prob_right], axis=1)

            # It uses Gumbel-max trick to draw samples from a discrete
            # distribution
            choosed_idx = xp.argmax(xp.random.gumbel(size=prob.shape) + prob,
                                    axis=1)
            columns = choosed_idx
            list_sampled_ids.append(node2word[start_ids, columns])
            list_next_ids.append(start_ids)
            next_ids = parent2child[start_ids][rows, columns]
            next_ids = xp.where(next_ids != LEAF, next_ids, FINISH_SAMPLING)

            # check whether all nodes are LEAF.
            if xp.all(next_ids == FINISH_SAMPLING):
                # if all nodes will reach leaf, then finish sampling.
                break
            start_ids = next_ids

        def xp_stack_func(x):
            return xp.reshape(xp.concatenate(x), (-1, batchsize)).T

        next_ids = xp_stack_func(list_next_ids)
        sampled_word_ids = xp_stack_func(list_sampled_ids)
        lengths = xp.argmax(next_ids == FINISH_SAMPLING, axis=1)
        max_length = next_ids.shape[1]
        lengths = xp.where(lengths == 0, max_length, lengths)

        output = sampled_word_ids[xp.arange(batchsize), lengths - 1]

        return output

    def __call__(self, x, t):
        """Computes the loss value for given input and ground truth labels.

        Args:
            x (~chainer.Variable): Input to the classifier at each node.
            t (~chainer.Variable): Batch of ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """
        f = copy.copy(self._func)  # creates a copy of the function node
        return f(x, t, self.W)
