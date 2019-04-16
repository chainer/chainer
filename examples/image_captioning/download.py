#!/usr/bin/env python
import argparse
import os
import zipfile

import progressbar
from six.moves.urllib import request

"""Download the MSCOCO dataset (images and captions)."""


urls = [
    'http://images.cocodataset.org/zips/train2014.zip',
    'http://images.cocodataset.org/zips/val2014.zip',
    'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
]


def download(url, dst_file_path):
    # Download a file, showing progress
    bar_wrap = [None]

    def reporthook(count, block_size, total_size):
        bar = bar_wrap[0]
        if bar is None:
            bar = progressbar.ProgressBar(
                maxval=total_size,
                widgets=[
                    progressbar.Percentage(),
                    ' ',
                    progressbar.Bar(),
                    ' ',
                    progressbar.FileTransferSpeed(),
                    ' | ',
                    progressbar.ETA(),
                ])
            bar.start()
            bar_wrap[0] = bar
        bar.update(min(count * block_size, total_size))

    request.urlretrieve(url, dst_file_path, reporthook=reporthook)


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
        download(url, dst_file_path)

        # Unzip the file
        zf = zipfile.ZipFile(dst_file_path)
        for name in zf.namelist():
            dirname, filename = os.path.split(name)
            if not filename == '':
                zf.extract(name, args.out)

        # Remove the zip file since it has been extracted
        os.remove(dst_file_path)
