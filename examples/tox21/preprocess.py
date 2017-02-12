import copy

from chainer import datasets as D
from rdkit.Chem import rdMolDescriptors


tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
               'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
               'SR-HSE', 'SR-MMP', 'SR-p53']


def default_preprocessor(mol_supplier):
    labels = label_extractor(copy.copy(mol_supplier))
    fvs = feature_extractor(mol_supplier)
    return D.TupleDataset(fvs, labels)


def label_extractor(mol_supplier):
    labels = []
    for mol in mol_supplier:
        if mol is None:
            continue
        label = []
        for task in tox21_tasks:
            if mol.HasProp(task):
                label.append(int(mol.GetProp(task)))
            else:
                label.append(-1)
        labels.append(label)
    return numpy.array(labels, dtype=numpy.int8)


def feature_extractor(mol_supplier, radius=2):
    fps = []
    for mol in mol_supplier:
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius)
        fps.append(fp)
    return numpy.array(fp, dtype=numpy.bool_)
