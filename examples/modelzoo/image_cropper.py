#!/usr/bin/env python
# coding:utf-8


import cv2
import numpy as np
import os
from PIL import Image

class ImageCropper(object):
    INITIAL_IMAGE_SIZE = 256
    SCALE_FACTOR = 255

    def __init__(self, in_size):
        self.in_size = in_size
        self.cropwidth = self.INITIAL_IMAGE_SIZE - self.in_size

    def crop(self, img, top, left):
        bottom = self.in_size + top
        right  = self.in_size + left
        return img[:, top:bottom, left:right].astype(np.float32)

    # test ok
    def crop_center_image(self, image, is_scaled=True):
        top  = self.cropwidth / 2
        left = self.cropwidth / 2
        image = self.crop(image, top, left)
        if is_scaled:
            image /= self.SCALE_FACTOR
        return image

    # test ok
    def crop_center(self, path, is_scaled=True):
        # Data loading routine
        # image = cv2.imread(path).transpose(2, 0, 1)
        image = np.asarray(Image.open(path).convert('RGB')).transpose(2, 0, 1)[::-1]
        return self.crop_center_image(image, is_scaled)
