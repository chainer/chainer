from chainer.dataset import download
import zipfile
import os
import request

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import SDMolSupplier
import numpy

train = "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf"
test = "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_testsdf"
validate = "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_scoresdf"

root = download.get_dataset_directory('pfnet/chainer/tox21')
path = os.path.join(root, "train")
download.cache_or_load_file(path, creator, loader)

def creator(path):
    train_path = download.cached_download(train)

    with zipfile.ZipFile(train_path, 'r') as z:
        z.extract('tox21_10k_challenge_test.sdf')
        fps = []
        labels = []
        for mol in SDMolSupplier('tox21_10k_challenge_test.sdf'):
            fp = rdMolDescriptor.GetMorganFingerprintAsBitVect(mol)
            label = [] # Get label from molcule object
            fps.append(fp)
            labels.append(label)

    # save cached file at path.
    fps = numpy.array(fp, dtype=numpy.bool_)
    labels = numpy.array(labels, dtype=numpy.bool_)
    numpy.savez_compressed(path, fps=fps, labels=labels)
    return {'fps': fps, 'labels': labels}

def loader(path):
    return numpy.load(path)
