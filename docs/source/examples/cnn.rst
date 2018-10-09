Convolutional Network for Visual Recognition Tasks
``````````````````````````````````````````````````

.. currentmodule:: chainer

In this section, you will learn how to write

* A small convolutional network with a model class that is inherited from :class:`~chainer.Chain`,
* A large convolutional network that has several building block networks with :class:`~chainer.ChainList`.

After reading this section, you will be able to:

* Write your own original convolutional network in Chainer

A convolutional network (ConvNet) is mainly comprised of convolutional layers.
This type of network is commonly used for various visual recognition tasks,
e.g., classifying hand-written digits or natural images into given object
classes, detecting objects from an image, and labeling all pixels of an image
with the object classes (semantic segmentation), and so on.

In such tasks, a typical ConvNet takes a set of images whose shape is
:math:`(N, C, H, W)`, where

- :math:`N` denotes the number of images in a mini-batch,
- :math:`C` denotes the number of channels of those images,
- :math:`H` and :math:`W` denote the height and width of those images,

respectively. Then, it typically outputs a fixed-sized vector as membership
probabilities over the target object classes. It also can output a set of
feature maps that have the corresponding size to the input image for a pixel
labeling task, etc.

.. include:: ../imports.rst

LeNet5
''''''

Here, let's start by defining LeNet5 [LeCun98]_ in Chainer.
This is a ConvNet model that has 5 layers comprised of 3 convolutional layers
and 2 fully-connected layers. This was proposed to classify hand-written
digit images in 1998. In Chainer, the model can be written as follows:

.. testcode::

    class LeNet5(Chain):
        def __init__(self):
            super(LeNet5, self).__init__()
            with self.init_scope():
                self.conv1 = L.Convolution2D(
                    in_channels=1, out_channels=6, ksize=5, stride=1)
                self.conv2 = L.Convolution2D(
                    in_channels=6, out_channels=16, ksize=5, stride=1)
                self.conv3 = L.Convolution2D(
                    in_channels=16, out_channels=120, ksize=4, stride=1)
                self.fc4 = L.Linear(None, 84)
                self.fc5 = L.Linear(84, 10)

        def forward(self, x):
            h = F.sigmoid(self.conv1(x))
            h = F.max_pooling_2d(h, 2, 2)
            h = F.sigmoid(self.conv2(h))
            h = F.max_pooling_2d(h, 2, 2)
            h = F.sigmoid(self.conv3(h))
            h = F.sigmoid(self.fc4(h))
            if chainer.config.train:
                return self.fc5(h)
            return F.softmax(self.fc5(h))

A typical way to write your network is creating a new class inherited from
:class:`~chainer.Chain` class. When defining your model in this way, typically,
all the layers which have trainable parameters are registered to the model
by assigning the objects of :class:`~chainer.Link` as an attribute.

The model class is instantiated before the forward and backward computations.
To give input images and label vectors simply by calling the model object
like a function, :meth:`forward` is usually defined in the model class.
This method performs the forward computation of the model. Chainer uses
the powerful autograd system for any computational graphs written with
:class:`~chainer.FunctionNode`\ s and :class:`~chainer.Link`\ s (actually a
:class:`~chainer.Link` calls a corresponding :class:`~chainer.FunctionNode`
inside of it), so that you don't need to explicitly write the code for backward
computations in the model. Just prepare the data, then give it to the model.
The way this works is the resulting output :class:`~chainer.Variable` from the
forward computation has a :meth:`~chainer.Variable.backward` method to perform
autograd. In the above model, :meth:`forward` has a ``if`` statement at the
end to switch its behavior by the Chainer's running mode, i.e., training mode or
not. Chainer presents the running mode as a global variable ``chainer.config.train``.
When it's in training mode, :meth:`forward` returns the output value of the
last layer as is to compute the loss later on, otherwise it returns a
prediction result by calculating :meth:`~chainer.functions.softmax`.

.. note::

  In Chainer v1, if a function or link behaved differently in
  training and other modes, it was common that it held an attribute
  that represented its running mode or was provided with the mode
  from outside as an argument. In Chainer v2, it is recommended to use
  the global configuration ``chainer.config.train`` to switch the running mode.

If you don't want to write ``conv1`` and the other layers more than once, you
can also write the same model like in this way:

.. testcode::

    from functools import partial

    class LeNet5(Chain):
        def __init__(self):
            super(LeNet5, self).__init__()
            net = [('conv1', L.Convolution2D(1, 6, 5, 1))]
            net += [('_sigm1', F.sigmoid)]
            net += [('_mpool1', partial(F.max_pooling_2d, ksize=2, stride=2))]
            net += [('conv2', L.Convolution2D(6, 16, 5, 1))]
            net += [('_sigm2', F.sigmoid)]
            net += [('_mpool2', partial(F.max_pooling_2d, ksize=2, stride=2))]
            net += [('conv3', L.Convolution2D(16, 120, 4, 1))]
            net += [('_sigm3', F.sigmoid)]
            net += [('_mpool3', partial(F.max_pooling_2d, ksize=2, stride=2))]
            net += [('fc4', L.Linear(None, 84))]
            net += [('_sigm4', F.sigmoid)]
            net += [('fc5', L.Linear(84, 10))]
            net += [('_sigm5', F.sigmoid)]
            with self.init_scope():
                for n in net:
                    if not n[0].startswith('_'):
                        setattr(self, n[0], n[1])
            self.layers = net

        def forward(self, x):
            for n, f in self.layers:
                if not n.startswith('_'):
                    x = getattr(self, n)(x)
                else:
                    x = f(x)
            if chainer.config.train:
                return x
            return F.softmax(x)

.. note::

    You can also use :class:`~chainer.Sequential` to write the above model
    more simply. Please note that :class:`~chainer.Sequential` is an
    experimental feature introduced in Chainer v4 and its interface may be
    changed in the future versions.

This code creates a list of pairs of component name (e.g., ``conv1``, ``_sigm1``, etc.) and all :class:`~chainer.Link`\ s and functions (e.g., ``F.sigmoid``, which internally invokes :class:`~chainer.FunctionNode`) after calling its superclass's constructor.
In this case, components whose name start with ``_`` are functions (:class:`~chainer.FunctionNode`), which doesn't have any trainable parameters, so that we don't register (``setattr``) it to the model.
Others (``conv1``, ``fc4``, etc.) are :class:`~chainer.Link`\ s, which are trainable layers that hold parameters.
This operation can be freely replaced with many other ways because those component names are just designed to select :class:`~chainer.Link`\ s only from the list ``net`` easily.
The list ``net`` is stored as an attribute :attr:`layers` to refer it in
:meth:`forward`. In :meth:`forward`, it retrieves all layers in the network
from :attr:`self.forward` sequentially and gives the
input variable or the intermediate output from the previous layer to the
current layer. The last part of the :meth:`forward` to switch its behavior
by the training/inference mode is the same as the former way.

Ways to calculate loss
......................

When you train the model with label vector ``t``, the loss should be calculated
using the output from the model. There also are several ways to calculate the
loss:

.. testcode::

    model = LeNet5()

    # Input data and label
    x = np.random.rand(32, 1, 28, 28).astype(np.float32)
    t = np.random.randint(0, 10, size=(32,)).astype(np.int32)

    # Forward computation
    y = model(x)

    # Loss calculation
    loss = F.softmax_cross_entropy(y, t)

This is a primitive way to calculate a loss value from the output of the model.
On the other hand, the loss computation can be included in the model itself by
wrapping the model object (:class:`~chainer.Chain` or
:class:`~chainer.ChainList` object) with a class inherited from
:class:`~chainer.Chain`. The outer :class:`~chainer.Chain` should take the
model defined above and register it with :meth:`~chainer.Chain.init_scope`.
:class:`~chainer.Chain` is actually
inherited from :class:`~chainer.Link`, so that :class:`~chainer.Chain` itself
can also be registered as a trainable :class:`~chainer.Link` to another
:class:`~chainer.Chain`. Actually, :class:`~chainer.links.Classifier` class to
wrap the model and add the loss computation to the model already exists.
Actually, there is already a :class:`~chainer.links.Classifier` class that can
be used to wrap the model and include the loss computation as well.
It can be used like this:

.. testcode::

    model = L.Classifier(LeNet5())

    # Foward & Loss calculation
    loss = model(x, t)

This class takes a model object as an input argument and registers it to
a ``predictor`` property as a trained parameter. As shown above, the returned
object can then be called like a function in which we pass ``x`` and ``t`` as
the input arguments and the resulting loss value (which we recall is a
:class:`~chainer.Variable`) is returned.

See the detailed implementation of :class:`~chainer.links.Classifier` from
here: :class:`chainer.links.Classifier` and check the implementation by looking
at the source.

From the above examples, we can see that Chainer provides the flexibility to
write our original network in many different ways. Such flexibility intends to
make it intuitive for users to design new and complex models.

VGG16
'''''

Next, let's write some larger models in Chainer. When you write a large network
consisting of several building block networks, :class:`~chainer.ChainList` is
useful. First, let's see how to write a VGG16 [Simonyan14]_ model.

.. testcode::


    class VGG16(chainer.ChainList):
        def __init__(self):
            super(VGG16, self).__init__(
                VGGBlock(64),
                VGGBlock(128),
                VGGBlock(256, 3),
                VGGBlock(512, 3),
                VGGBlock(512, 3, True))

        def forward(self, x):
            for f in self.children():
                x = f(x)
            if chainer.config.train:
                return x
            return F.softmax(x)


    class VGGBlock(chainer.Chain):
        def __init__(self, n_channels, n_convs=2, fc=False):
            w = chainer.initializers.HeNormal()
            super(VGGBlock, self).__init__()
            with self.init_scope():
                self.conv1 = L.Convolution2D(None, n_channels, 3, 1, 1, initialW=w)
                self.conv2 = L.Convolution2D(
                    n_channels, n_channels, 3, 1, 1, initialW=w)
                if n_convs == 3:
                    self.conv3 = L.Convolution2D(
                        n_channels, n_channels, 3, 1, 1, initialW=w)
                if fc:
                    self.fc4 = L.Linear(None, 4096, initialW=w)
                    self.fc5 = L.Linear(4096, 4096, initialW=w)
                    self.fc6 = L.Linear(4096, 1000, initialW=w)

            self.n_convs = n_convs
            self.fc = fc

        def forward(self, x):
            h = F.relu(self.conv1(x))
            h = F.relu(self.conv2(h))
            if self.n_convs == 3:
                h = F.relu(self.conv3(h))
            h = F.max_pooling_2d(h, 2, 2)
            if self.fc:
                h = F.dropout(F.relu(self.fc4(h)))
                h = F.dropout(F.relu(self.fc5(h)))
                h = self.fc6(h)
            return h

That's it. VGG16 is a model which won the 1st place in
`classification + localization task at ILSVRC 2014 <http://www.image-net.org/challenges/LSVRC/2014/results#clsloc>`_,
and since then, has become one of the standard models for many different tasks
as a pre-trained model. This has 16-layers, so it's called "VGG-16", but we can
write this model without writing all layers independently. Since this model
consists of several building blocks that have the same architecture, we can
build the whole network by re-using the building block definition. Each part
of the network is consisted of 2 or 3 convolutional layers and activation
function (:meth:`~chainer.functions.relu`) following them, and
:meth:`~chainer.functions.max_pooling_2d` operations. This block is written as
:class:`VGGBlock` in the above example code. And the whole network just calls
this block one by one in sequential manner.

ResNet152
'''''''''

How about ResNet? ResNet [He16]_ came in the following year's ILSVRC. It is a
much deeper model than VGG16, having up to 152 layers. This sounds super
laborious to build, but it can be implemented in almost same manner as VGG16.
In the other words, it's easy. One possible way to write ResNet-152 is:

.. testcode::

    class ResNet152(chainer.Chain):
        def __init__(self, n_blocks=[3, 8, 36, 3]):
            w = chainer.initializers.HeNormal()
            super(ResNet152, self).__init__()
            with self.init_scope():
                self.conv1 = L.Convolution2D(None, 64, 7, 2, 3, initialW=w, nobias=True)
                self.bn1 = L.BatchNormalization(64)
                self.res2 = ResBlock(n_blocks[0], 64, 64, 256, 1)
                self.res3 = ResBlock(n_blocks[1], 256, 128, 512)
                self.res4 = ResBlock(n_blocks[2], 512, 256, 1024)
                self.res5 = ResBlock(n_blocks[3], 1024, 512, 2048)
                self.fc6 = L.Linear(2048, 1000)

        def forward(self, x):
            h = self.bn1(self.conv1(x))
            h = F.max_pooling_2d(F.relu(h), 2, 2)
            h = self.res2(h)
            h = self.res3(h)
            h = self.res4(h)
            h = self.res5(h)
            h = F.average_pooling_2d(h, h.shape[2:], stride=1)
            h = self.fc6(h)
            if chainer.config.train:
                return h
            return F.softmax(h)


    class ResBlock(chainer.ChainList):
        def __init__(self, n_layers, n_in, n_mid, n_out, stride=2):
            super(ResBlock, self).__init__()
            self.add_link(BottleNeck(n_in, n_mid, n_out, stride, True))
            for _ in range(n_layers - 1):
                self.add_link(BottleNeck(n_out, n_mid, n_out))

        def forward(self, x):
            for f in self.children():
                x = f(x)
            return x


    class BottleNeck(chainer.Chain):
        def __init__(self, n_in, n_mid, n_out, stride=1, proj=False):
            w = chainer.initializers.HeNormal()
            super(BottleNeck, self).__init__()
            with self.init_scope():
                self.conv1x1a = L.Convolution2D(
                    n_in, n_mid, 1, stride, 0, initialW=w, nobias=True)
                self.conv3x3b = L.Convolution2D(
                    n_mid, n_mid, 3, 1, 1, initialW=w, nobias=True)
                self.conv1x1c = L.Convolution2D(
                    n_mid, n_out, 1, 1, 0, initialW=w, nobias=True)
                self.bn_a = L.BatchNormalization(n_mid)
                self.bn_b = L.BatchNormalization(n_mid)
                self.bn_c = L.BatchNormalization(n_out)
                if proj:
                    self.conv1x1r = L.Convolution2D(
                        n_in, n_out, 1, stride, 0, initialW=w, nobias=True)
                    self.bn_r = L.BatchNormalization(n_out)
            self.proj = proj

        def forward(self, x):
            h = F.relu(self.bn_a(self.conv1x1a(x)))
            h = F.relu(self.bn_b(self.conv3x3b(h)))
            h = self.bn_c(self.conv1x1c(h))
            if self.proj:
                x = self.bn_r(self.conv1x1r(x))
            return F.relu(h + x)

In the :class:`BottleNeck` class, depending on the value of the proj argument
supplied to the initializer, it will conditionally compute a convolutional
layer ``conv1x1r`` which will extend the number of channels of the input ``x``
to be equal to the number of channels of the output of ``conv1x1c``, and
followed by a batch normalization layer before the final ReLU layer.
Writing the building block in this way improves the re-usability of a class.
It switches not only the behavior in :meth:`__class__` by flags but also the
parameter registration. In this case, when :attr:`proj` is ``False``, the
:class:`BottleNeck` doesn't have `conv1x1r` and `bn_r` layers, so the memory
usage would be efficient compared to the case when it registers both anyway and
just ignore them if :attr:`proj` is ``False``.

Using nested :class:`~chainer.Chain`\ s and :class:`~chainer.ChainList` for
sequential part enables us to write complex and very deep models easily.

Use Pre-trained Models
''''''''''''''''''''''

Various ways to write your models were described above. It turns out that
VGG16 and ResNet are very useful as general feature extractors for many kinds
of tasks, including but not limited to image classification. So, Chainer
provides you with the pre-trained VGG16 and ResNet-50/101/152 models with a
simple API. You can use these models as follows:

.. testcode::

    from chainer.links import VGG16Layers

    model = VGG16Layers()

When :class:`~chainer.links.VGG16Layers` is instantiated, the pre-trained
parameters are automatically downloaded from the author's server. So you can
immediately start to use VGG16 with pre-trained weight as a good image feature
extractor. See the details of this model here:
:class:`chainer.links.VGG16Layers`.

In the case of ResNet models, there are three variations differing in the number
of layers. We have :class:`chainer.links.ResNet50Layers`,
:class:`chainer.links.ResNet101Layers`, and :class:`chainer.links.ResNet152Layers` models
with easy parameter loading feature. ResNet's pre-trained parameters are not
available for direct downloading, so you need to download the weight from the
author's web page first, and then place it into the dir
``$CHAINER_DATSET_ROOT/pfnet/chainer/models`` or your favorite place. Once
the preparation is finished, the usage is the same as VGG16:

.. testcode::

    from chainer.links import ResNet152Layers

    model = ResNet152Layers()

.. testoutput::
   :options: -IGNORE_EXCEPTION_DETAIL

   Traceback (most recent call last):
   OSError: The pre-trained caffemodel does not exist. Please download it from 'https://github.com/KaimingHe/deep-residual-networks', and place it on ...

Please see the details of usage and how to prepare the pre-trained weights for
ResNet here: :class:`chainer.links.ResNet50Layers`

References
..........

.. [LeCun98] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.
    Gradient-based learning applied to documentation recognition. Proceedings of the
    IEEE, 86(11), 2278–2324, 1998.
.. [Simonyan14] Simonyan, K. and Zisserman, A., Very Deep Convolutional
    Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556,
    2014.
.. [He16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual
    Learning for Image Recognition. The IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), pp. 770-778, 2016.
