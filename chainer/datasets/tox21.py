import os
import shutil
import warnings
import zipfile

import numpy

from chainer.dataset import download
from chainer.datasets import tuple_dataset

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    available = True
except Exception as e:
    available = False
    _resolution_error = e


def check_available():
    """Checks availability of Tox21 dataset.

    This function checks the availability of Tox21 dataset
    in user's environment.
    Specifically, we use `RDKit <https://github.com/rdkit/rdkit>`_
    to extract features and labels from raw files, whose format are
    `SDF <https://en.wikipedia.org/wiki/Chemical_table_file#SDF>`_.
    So, it raises error when Chainer failed to import RDKit module.

    Returns:
        None

    """
    if not available:
        msg = ('rdkit is not installed in your environment. '
               'Please install it to use tox21 dataset.\n'
               'See the official document for installation. '
               'http://www.rdkit.org/docs/Install.html')
        msg += str(_resolution_error)
        raise RuntimeError(msg)


_config = {
    'train': {
        'url': 'https://tripod.nih.gov/tox21/challenge/download?'
        'id=tox21_10k_data_allsdf',
        'filename': 'tox21_10k_data_all.sdf'
    },
    'val': {
        'url': 'https://tripod.nih.gov/tox21/challenge/download?'
        'id=tox21_10k_challenge_testsdf',
        'filename': 'tox21_10k_challenge_test.sdf'
    },
    'test': {
        'url': 'https://tripod.nih.gov/tox21/challenge/download?'
        'id=tox21_10k_challenge_scoresdf',
        'filename': 'tox21_10k_challenge_score.sdf'
    }
}

_root = 'pfnet/chainer/tox21'

label_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
               'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
               'SR-HSE', 'SR-MMP', 'SR-p53']


def _ECFP(mol_supplier, radius=2):
    descriptors = []
    labels = []
    for mol in mol_supplier:
        if mol is None:
            continue
        label = []
        for task in label_names:
            if mol.HasProp(task):
                label.append(int(mol.GetProp(task)))
            else:
                label.append(-1)
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius)
        except Exception:
            continue
        descriptors.append(fp)
        labels.append(label)
    descriptors = numpy.array(descriptors, dtype=numpy.float32)
    labels = numpy.array(labels, dtype=numpy.int32)
    return descriptors, labels


def _creator(cached_file_path, sdffile, url):
    download_file_path = download.cached_download(url)

    with zipfile.ZipFile(download_file_path, 'r') as z:
        z.extract(sdffile)
    mol_supplier = Chem.SDMolSupplier(sdffile)
    shutil.move(sdffile, cached_file_path)
    return mol_supplier


def _get_tox21(config_name, preprocessor, with_label=True):
    basename = config_name
    c = _config[config_name]
    url = c['url']
    sdffile = c['filename']

    cache_root = download.get_dataset_directory(_root)
    cache_path = os.path.join(cache_root, basename + ".sdf")

    def creator(path):
        return _creator(path, sdffile, url)

    mol_supplier = download.cache_or_load_file(
        cache_path, creator, Chem.SDMolSupplier)

    descriptors, labels = preprocessor(mol_supplier)
    if with_label:
        return tuple_dataset.TupleDataset(descriptors, labels)
    else:
        return descriptors


def get_tox21(preprocessor=_ECFP):
    """Downloads, caches and preprocesses Tox21 dataset.

    Args:
        preprocessor: A module used for preprocessing of
            raw files.
            It should be a callable which takes an iterable of
            :class:`rdkit.Chem.rdchem.Mol` objects and a tuple
            of strings each of which represents a label name.
            It should return an instance of
            :class:`chainer.datasets.TupleDataset` that
            represents a pair of feature vectors and labels.

    Returns:
        The 3-tuple consisting of train, validation and test
        datasets, respectively. The train and validation
        datasets are pairs of descriptors and labels.
        The test dataset only has descriptors and does
        not have labels.

    """

    check_available()
    train = _get_tox21('train', preprocessor)
    val = _get_tox21('val', preprocessor)
    test = _get_tox21('test', preprocessor, False)
    return train, val, test
