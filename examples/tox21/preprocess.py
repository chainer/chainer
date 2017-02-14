from chainer import datasets as D
import numpy
from rdkit.Chem import rdMolDescriptors


tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
               'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
               'SR-HSE', 'SR-MMP', 'SR-p53']


def ECFP(mol_supplier, radius=2):
    fps = []
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
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius)
        except Exception as e:
            print(e)
            continue
        fps.append(fp)
        labels.append(label)
    fps = numpy.array(fps, dtype=numpy.float32)
    labels = numpy.array(labels, dtype=numpy.int32)
    assert len(fps) == len(labels)
    return D.TupleDataset(fps, labels)
