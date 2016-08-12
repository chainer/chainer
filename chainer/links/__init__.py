"""Collection of :class:`~chainer.Link` implementations."""

from chainer.links.activation import maxout
from chainer.links.activation import prelu
from chainer.links.connection import bias
from chainer.links.connection import bilinear
from chainer.links.connection import convolution_2d
from chainer.links.connection import deconvolution_2d
from chainer.links.connection import embed_id
from chainer.links.connection import gru
from chainer.links.connection import inception
from chainer.links.connection import inceptionbn
from chainer.links.connection import linear
from chainer.links.connection import lstm
from chainer.links.connection import mgu
from chainer.links.connection import mlp_convolution_2d
from chainer.links.connection import parameter
from chainer.links.connection import peephole
from chainer.links.connection import scale
from chainer.links.connection import sgu
from chainer.links.loss import crf1d
from chainer.links.loss import hierarchical_softmax
from chainer.links.loss import negative_sampling
from chainer.links.model import classifier
from chainer.links.normalization import batch_normalization


Maxout = maxout.Maxout
PReLU = prelu.PReLU

Bias = bias.Bias
Bilinear = bilinear.Bilinear
Convolution2D = convolution_2d.Convolution2D
Deconvolution2D = deconvolution_2d.Deconvolution2D
EmbedID = embed_id.EmbedID
GRU = gru.GRU
StatefulGRU = gru.StatefulGRU
Inception = inception.Inception
InceptionBN = inceptionbn.InceptionBN
Linear = linear.Linear
LSTM = lstm.LSTM
StatelessLSTM = lstm.StatelessLSTM
MGU = mgu.MGU
StatefulMGU = mgu.StatefulMGU
MLPConvolution2D = mlp_convolution_2d.MLPConvolution2D
Parameter = parameter.Parameter
PeepholeLSTM = peephole.PeepholeLSTM
StatefulPeepholeLSTM = peephole.StatefulPeepholeLSTM
Scale = scale.Scale
SGU = sgu.SGU
StatefulSGU = sgu.StatefulSGU
DSGU = sgu.DSGU
StatefulDSGU = sgu.StatefulDSGU
Scale = scale.Scale

CRF1d = crf1d.CRF1d
BinaryHierarchicalSoftmax = hierarchical_softmax.BinaryHierarchicalSoftmax
NegativeSampling = negative_sampling.NegativeSampling

Classifier = classifier.Classifier

BatchNormalization = batch_normalization.BatchNormalization
