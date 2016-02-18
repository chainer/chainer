import random


def crop_center(image, cropsize):
    """Crops the center of given array.

    TODO(beam2d): document it.

    """
    if image.ndim < len(cropsize):
        raise ValueError('image shape and crop size mismatched')
    if image.ndim > len(cropsize):
        cropsize = image.shape[:-len(cropsize)] + tuple(cropsize)

    slices = []
    for dim, crop in zip(image.shape, cropsize):
        top = (dim - crop) // 2
        bottom = top + crop
        slices.append(slice(top, bottom))

    return image[slices]


def crop_random(image, cropsize):
    """Crops a randomly chosen range of given array.

    TODO(beam2d): document it.

    """
    if image.ndim < len(cropsize):
        raise ValueError('image shape and crop size mismatched')
    if image.ndim > len(cropsize):
        cropsize = image.shape[:-len(cropsize)] + tuple(cropsize)

    slices = []
    for dim, crop in zip(image.shape, cropsize):
        top = random.randint(0, dim - crop)
        bottom = top + crop
        slices.append(slice(top, bottom))

    return image[slices]
