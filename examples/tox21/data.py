import os
import shutil
import zipfile

from chainer.dataset import download
from rdkit import Chem

import preprocess


config = {
    'train': {
        'url': 'https://tripod.nih.gov/tox21/challenge/download?'
        'id=tox21_10k_data_allsdf',
        'filename': 'tox21_10k_data_all.sdf'
        },
    'test': {
        'url': 'https://tripod.nih.gov/tox21/challenge/download?'
        'id=tox21_10k_challenge_testsdf',
        'filename': 'tox21_10k_challenge_test.sdf'
        },
    'val': {
        'url': 'https://tripod.nih.gov/tox21/challenge/download?'
        'id=tox21_10k_challenge_scoresdf',
        'filename': 'tox21_10k_challenge_score.sdf'
        }
    }

root = 'pfnet/chainer/tox21'


def _creator(cached_file_path, sdffile, url):
    download_file_path = download.cached_download(url)

    with zipfile.ZipFile(download_file_path, 'r') as z:
        z.extract(sdffile)
    mol_supplier = Chem.SDMolSupplier(sdffile)
    shutil.move(sdffile, cached_file_path)
    return mol_supplier


def _loader(path):
    return Chem.SDMolSupplier(path)


def _get_tox21(config_name, preprocessor=preprocess.ECFP):
    basename = config_name
    global config
    c = config[config_name]
    url = c['url']
    sdffile = c['filename']

    cache_root = download.get_dataset_directory(root)
    cache_path = os.path.join(cache_root, basename + ".sdf")

    def creator(path):
        return _creator(path, sdffile, url)

    mol_supplier = download.cache_or_load_file(
        cache_path, creator, _loader)
    return preprocessor(mol_supplier)


def get_tox21(preprocessor=preprocess.ECFP):
    train = _get_tox21('train', preprocessor)
    test = _get_tox21('test', preprocessor)
    val = _get_tox21('val', preprocessor)
    return train, test, val
