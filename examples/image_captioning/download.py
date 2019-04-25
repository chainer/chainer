#!/usr/bin/env python
import argparse
import os
from six.moves.urllib import request
import zipfile


"""Download the MSCOCO dataset (images and captions)."""


urls = [
    'http://images.cocodataset.org/zips/train2014.zip',
    'http://images.cocodataset.org/zips/val2014.zip',
    'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='data',
                        help='Target MSOCO dataset root directory')
    args = parser.parse_args()

    try:
        os.makedirs(args.out)
    except OSError:
        raise OSError(
            '\'{}\' already exists, delete it and try again'.format(args.out))

    for url in urls:
        print('Downloading {}...'.format(url))

        # Download the zip file
        file_name = os.path.basename(url)
        dst_file_path = os.path.join(args.out, file_name)
        request.urlretrieve(url, dst_file_path)

        # Unzip the file
        zf = zipfile.ZipFile(dst_file_path)
        for name in zf.namelist():
            dirname, filename = os.path.split(name)
            if not filename == '':
                zf.extract(name, args.out)

        # Remove the zip file since it has been extracted
        os.remove(dst_file_path)
