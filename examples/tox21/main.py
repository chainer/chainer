from chainer.dataset import download
import zipfile
import os

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import SDMolSupplier
import numpy


train = "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf"
test = "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_testsdf"
validate = "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_scoresdf"

tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
               'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
               'SR-HSE', 'SR-MMP', 'SR-p53']

def preprocess(sdf):
    fps = []
    labels = []
    sdmol = SDMolSupplier('tox21_10k_data_all.sdf')
    for mol in sdmol:
        if mol is None:
            continue
        label = []
        for task in tox21_tasks:
            if mol.HasProp(task):
                label.append(int(mol.GetProp(task)))
            else:
                label.append(-1)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
        fps.append(fp)
        labels.append(label)
    fps = numpy.array(fp, dtype=numpy.bool_)
    labels = numpy.array(labels, dtype=numpy.bool_)
    return {'fps': fps, 'labels': labels}
    

def creator(path):
    train_path = download.cached_download(train)

    with zipfile.ZipFile(train_path, 'r') as z:
        z.extract('tox21_10k_data_all.sdf')
        fps, labels = preprocess(sdf)
    numpy.savez_compressed(path, fps=fps, labels=labels)
    return {'fps': fps, 'labels': labels}


def loader(path):
    return numpy.load(path)

root = download.get_dataset_directory('pfnet/chainer/tox21')
path = os.path.join(root, "train.npz")
train_dataset = download.cache_or_load_file(path, creator, loader)
print(type(train_dataset))


