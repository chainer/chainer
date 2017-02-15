import os
import shutil
import warnings
import zipfile


from chainer.dataset import download
from chainer import datasets as D
import numpy

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    _available = True
except ImportError:
    _available = False


def check_available():
    if not _available:
        warnings.warn('rdkit is not install on your environment '
                      'Please install it to use tox21 dataset.\n'
                      'See the official document for installation.'
                      'http://www.rdkit.org/docs/Install.html')
    return _available


config = {
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

root = 'pfnet/chainer/tox21'

tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
               'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
               'SR-HSE', 'SR-MMP', 'SR-p53']


def _ECFP(mol_supplier, radius=2, label_names=tox21_tasks):
    fps = []
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
        except Exception as e:
            print(e)
            continue
        fps.append(fp)
        labels.append(label)
    fps = numpy.array(fps, dtype=numpy.float32)
    labels = numpy.array(labels, dtype=numpy.int32)
    if label_names:
        assert len(fps) == len(labels)
        return D.TupleDataset(fps, labels)
    else:
        return fps


def _creator(cached_file_path, sdffile, url):
    download_file_path = download.cached_download(url)

    with zipfile.ZipFile(download_file_path, 'r') as z:
        z.extract(sdffile)
    mol_supplier = Chem.SDMolSupplier(sdffile)
    shutil.move(sdffile, cached_file_path)
    return mol_supplier


def _get_tox21(config_name, preprocessor):
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
        cache_path, creator, Chem.SDMolSupplier)
    return preprocessor(mol_supplier)


def get_tox21(preprocessor=_ECFP):
    if check_available():
        train = _get_tox21('train', preprocessor)
        val = _get_tox21('val', preprocessor)
        test = _get_tox21('test', preprocessor)
        return train, val, test
