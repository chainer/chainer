import numpy as np

import chainer
from chainer import functions as F
from chainer import initializers
from chainer import links as L
from chainer import reporter
from chainer import Variable


class ImageCaptionModel(chainer.Chain):

    """Image captioning model."""

    def __init__(self, vocab_size, hidden_size=512, rnn='lstm',
                 dropout_ratio=0.5, finetune_feat_extractor=False,
                 ignore_label=-1):
        super(ImageCaptionModel, self).__init__()

        if rnn == 'lstm':
            LanguageModel = LSTMLanguageModel
        elif rnn == 'nsteplstm':
            LanguageModel = NStepLSTMLanguageModel
        else:
            raise ValueError('Invalid RNN type.')

        with self.init_scope():
            self.feat_extractor = VGG16FeatureExtractor()
            self.lang_model = LanguageModel(
                vocab_size, hidden_size, dropout_ratio=dropout_ratio,
                ignore_label=ignore_label)

        self.finetune_feat_extractor = finetune_feat_extractor

    def prepare(self, img):
        """Single image to resized and normalized image."""
        return self.feat_extractor.prepare(img)

    def forward(self, imgs, captions):
        """Batch of images to a single loss."""
        imgs = Variable(imgs)
        if self.finetune_feat_extractor:
            img_feats = self.feat_extractor(imgs)
        else:
            # Extract features with the `train` configuration set to `False` in
            # order to basically skip the dropout regularizations. This is how
            # dropout is used during standard inference. Also, since we are not
            # going to optimize the feature extractor, we explicitly set the
            # backpropgation mode to not construct any computational graphs.
            with chainer.using_config('train', False), \
                    chainer.no_backprop_mode():
                img_feats = self.feat_extractor(imgs)

        loss = self.lang_model(img_feats, captions)

        # Report the loss so that it can be printed, logged and plotted by
        # other trainer extensions
        reporter.report({'loss': loss}, self)

        return loss

    def predict(self, imgs, bos, eos, max_caption_length):
        """Batch of images to captions."""
        imgs = Variable(imgs)
        img_feats = self.feat_extractor(imgs)
        captions = self.lang_model.predict(
            img_feats, bos=bos, eos=eos, max_caption_length=max_caption_length)
        return captions


class VGG16FeatureExtractor(chainer.Chain):

    """VGG16 image feature extractor."""

    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        with self.init_scope():
            self.cnn = L.VGG16Layers()
        self.cnn_layer_name = 'fc7'

    def prepare(self, img):
        """Single image to resized and normalized image.

        The inputs image should have the shape (channel, height, width) where
        channels are aligned RGB. The returned image has the same shape but
        channels in BGR order.
        """
        return L.model.vision.vgg.prepare(img)

    def forward(self, imgs):
        """Batch of images to image features."""
        img_feats = self.cnn(imgs, [self.cnn_layer_name])[self.cnn_layer_name]
        return img_feats


class LSTMLanguageModel(chainer.Chain):

    """Recurrent LSTM language model.

    Generate captions given features extracted from images.
    """

    def __init__(self, vocab_size, hidden_size, dropout_ratio, ignore_label):
        super(LSTMLanguageModel, self).__init__()
        with self.init_scope():
            self.embed_word = L.EmbedID(
                vocab_size,
                hidden_size,
                initialW=initializers.Normal(1.0),
                ignore_label=ignore_label
            )
            self.embed_img = L.Linear(
                hidden_size,
                initialW=initializers.Normal(0.01)
            )
            self.lstm = L.LSTM(hidden_size, hidden_size)
            self.out_word = L.Linear(
                hidden_size,
                vocab_size,
                initialW=initializers.Normal(0.01)
            )

        self.dropout_ratio = dropout_ratio

    def forward(self, img_feats, captions):
        """Batch of image features and image captions to a singe loss.

        Compute the softmax cross-entropy captioning loss.
        """
        self.reset(img_feats)

        loss = 0
        size = 0
        caption_length = captions.shape[1]
        for i in range(caption_length - 1):
            # Compute the loss based on the prediction of the next token in the
            # sequence
            x = Variable(self.xp.asarray(captions[:, i]))
            t = Variable(self.xp.asarray(captions[:, i + 1]))
            if (t.array == self.embed_word.ignore_label).all():
                # Preprocessed captions are padded to reach a maximum length.
                # Often, you want to set the `ignore_label` to this padding.
                # If all targets are simply paddings, predictions are no longer
                # required.
                break
            y = self.step(x)
            loss += F.softmax_cross_entropy(
                y, t, ignore_label=self.embed_word.ignore_label)
            size += 1
        return loss / max(size, 1)

    def predict(self, img_feats, bos, eos, max_caption_length):
        """Batch of image features to captions."""
        self.reset(img_feats)

        captions = self.xp.full((img_feats.shape[0], 1), bos, dtype=np.int32)
        for _ in range(max_caption_length):
            x = Variable(captions[:, -1])  # Previous word token as input
            y = self.step(x)
            pred = y.array.argmax(axis=1).astype(np.int32)
            captions = self.xp.hstack((captions, pred[:, None]))
            if (pred == eos).all():
                break
        return captions

    def reset(self, img_feats):
        """Batch of image features to hidden representations.

        Also, reset and then update the internal state of the LSTM.
        """
        self.lstm.reset_state()
        h = self.embed_img(img_feats)
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio))
        return h

    def step(self, x):
        """Batch of word tokens to word tokens.

        Predict the next set of tokens given previous tokens.
        """
        h = self.embed_word(x)
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio))
        h = self.out_word(F.dropout(h, ratio=self.dropout_ratio))
        return h


class NStepLSTMLanguageModel(chainer.Chain):
    """Recurrent NStepLSTM language model.

    Generate captions given features extracted from images.
    """

    def __init__(self, vocab_size, hidden_size, dropout_ratio, ignore_label):
        super(NStepLSTMLanguageModel, self).__init__()
        with self.init_scope():
            self.embed_word = L.EmbedID(
                vocab_size,
                hidden_size,
                initialW=initializers.Normal(1.0),
                ignore_label=ignore_label
            )
            self.embed_img = L.Linear(
                hidden_size,
                initialW=initializers.Normal(0.01)
            )
            self.lstm = L.NStepLSTM(1, hidden_size, hidden_size, dropout_ratio)
            self.decode_caption = L.Linear(
                hidden_size,
                vocab_size,
                initialW=initializers.Normal(0.01)
            )

        self.dropout_ratio = dropout_ratio

    def forward(self, img_feats, captions):
        """Batch of image features and image captions to a singe loss.

        Compute the softmax cross-entropy captioning loss in a single pass
        without iterating over the sequences.
        """
        hx, cx, _ = self.reset(img_feats)

        # Extract all inputs and targets for all captions in the batch
        xs = [c[:-1] for c in captions]  # del eos
        ts = [c[1:] for c in captions]  # del bos

        # Get the predictions `ys`
        _, _, ys = self.step(hx, cx, xs)

        # Since `ys` is concatenated, we also concatenate the target tokens
        # before computing the loss
        ts = F.concat(ts, axis=0)

        loss = F.softmax_cross_entropy(ys, ts)
        return loss

    def predict(self, img_feats, bos, eos, max_caption_length):
        """Batch of image features to captions."""
        hx, cx, _ = self.reset(img_feats)

        with chainer.using_device(self.device):
            xp = self.xp
            captions = xp.full(
                (img_feats.shape[0], 1), bos, dtype=np.int32)
            for i in range(max_caption_length):
                # Create a list of the previous tokens to treat as inputs
                xs = [xp.atleast_1d(c[-1]) for c in captions]

                # Get the predictions `ys`
                hx, cx, ys = self.step(hx, cx, xs)

                # From `ys`, get the indices for the highest confidence.
                # These indices correspond to the predicted tokens
                #
                # Note that this is a greedy approach and that it can by
                # replaced by e.g. beam search
                pred = ys.array.argmax(axis=1).astype(np.int32)
                captions = xp.hstack((captions, pred[:, None]))

                if (pred == eos).all():
                    break
        return captions

    def reset(self, img_feats):
        """Batch of image features to LSTM states and hidden representations.

        """
        h = self.embed_img(img_feats)
        h = F.split_axis(h, h.shape[0], axis=0)
        hx, cx, ys = self.lstm(None, None, h)
        return hx, cx, ys

    def step(self, hx, cx, xs):
        """Batch of word tokens to word tokens and hidden LSTM states.

        Predict the next set of tokens given previous tokens.
        """
        # Concatenate all input captions and pass them through the model in a
        # single pass
        caption_lens = [len(x) for x in xs]
        caption_sections = np.cumsum(caption_lens[:-1])
        xs = F.concat(xs, axis=0)

        xs = self.embed_word(xs)
        xs = F.split_axis(xs, caption_sections, axis=0)
        hx, cx, ys = self.lstm(hx, cx, xs)

        ys = F.concat(ys, axis=0)
        ys = F.dropout(ys, self.dropout_ratio)
        ys = self.decode_caption(ys)
        return hx, cx, ys
