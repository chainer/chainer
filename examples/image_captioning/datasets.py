from collections import defaultdict
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from chainer import dataset
from chainer.dataset.convert import to_device


# Vocabulary tokens of BOS (beginning of sentence), EOS (end of sentence),
# UNK (unknown word) and token labels to be ignored in the loss computation by
# the LSTM layers
_bos = 0
_eos = 1
_unk = 2
_ignore = -1


def split(sentence):
    return sentence.lower().replace('.', ' .').replace(',', ' ,').split()


class MsCocoDataset(dataset.DatasetMixin):

    """Wraps the MSCOCO datasets and is used by the iterator to fetch data."""

    def __init__(self, root_dir, data_dir, anno_file):
        coco = COCO(os.path.join(root_dir, anno_file))
        anns = coco.loadAnns(coco.getAnnIds())

        self.coco = coco
        self.anns = anns
        self.vocab = None  # Later set from outside
        self.coco_root = root_dir
        self.coco_data = data_dir

    def __len__(self):
        return len(self.anns)

    def get_example(self, i):
        """Called by the iterator to fetch a data sample.

        A data sample from MSCOCO consists of an image and its corresponding
        caption.

        The returned image has the shape (channel, height, width).
        """
        ann = self.anns[i]

        # Load the image
        img_id = ann['image_id']
        img_file_name = self.coco.loadImgs([img_id])[0]['file_name']
        img = Image.open(
            os.path.join(self.coco_root, self.coco_data, img_file_name))
        if img.mode == 'RGB':
            img = np.asarray(img, np.float32).transpose(2, 0, 1)
        elif img.mode == 'L':
            img = np.asarray(img, np.float32)
            img = np.broadcast_to(img, (3,) + img.shape)
        else:
            raise ValueError('Invalid image mode {}'.format(img.mode))

        # Load the caption, i.e. sequence of tokens
        tokens = [self.vocab.get(w, _unk) for w in
                  ['<bos>'] + split(ann['caption']) + ['<eos>']]
        tokens = np.array(tokens, np.int32)

        return img, tokens


def get_mscoco(
        root_dir,
        train_dir='train2014',
        train_anno='annotations/captions_train2014.json',
        val_dir='val2014',
        val_anno='annotations/captions_val2014.json',
        unk_threshold=5):
    """Return the training and validation datasets for MSCOCO.

    The datasets can be used by the iterator during training.

    A vocabulary is dynamically created based on all captions and is
    returned as members of the training and validation dataset objects.
    """
    train = MsCocoDataset(root_dir, train_dir, train_anno)
    val = MsCocoDataset(root_dir, val_dir, val_anno)

    # Create a vocabulary based on the captions from the training set only
    # (excluding the validation sets). This is common practice.
    captions = [ann['caption'] for ann in train.anns]

    # Filter out rare words as UNK
    word_counts = defaultdict(int)
    for c in captions:
        for w in split(c):
            word_counts[w] += 1

    # This vocabulary is needed in order to convert the words in the captions
    # to integer tokens. When generating captions during testing, these tokens
    # are mapped back to their corresponding words. Note that this vocabulary
    # sorted alphanumerically.
    vocab = {'<bos>': _bos, '<eos>': _eos, '<unk>': _unk}
    for w, count in sorted(word_counts.items()):
        if w not in vocab and count >= unk_threshold:
            vocab[w] = len(vocab)

    train.vocab = vocab
    val.vocab = vocab

    return train, val


def converter(batch, device, max_caption_length=None):
    """Optional preprocessing of the batch before forward pass."""
    pad = max_caption_length is not None

    imgs = []
    captions = []
    for img, caption in batch:
        # Preproess the caption by either fixing the length by padding (LSTM)
        # or by simply wrapping each caption in an ndarray (NStepLSTM)
        if pad:
            arr = np.full(max_caption_length, _ignore, dtype=np.int32)

            # Clip to max length if necessary
            arr[:len(caption)] = caption[:max_caption_length]
            caption = arr
        else:
            caption = to_device(device, np.asarray(caption, dtype=np.int32))

        imgs.append(img)
        captions.append(caption)

    if pad:
        captions = to_device(device, np.stack(captions))
    imgs = to_device(device, np.stack(imgs))

    return imgs, captions
