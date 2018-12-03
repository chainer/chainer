# Serialization

This example intends to show how to use serializers with minimal code.
These scripts show two ways to save/load snapshots using NumPy and H5Py, so please ensure that you have installed `h5py` package besides NumPy preliminary.

```bash
pip install h5py
```

## Save

```
python save.py
```

It creates `model.npz` and `model.h5` which store parameters of the same model. The above command shows the messages as follows:

```bash
model.npz saved!

--- The list of saved params in model.npz ---
persistent      : (10, 10)
l3/b    : (10,)
l3/W    : (10, 100)
l1/b    : (100,)
l1/W    : (100, 784)
l2/b    : (100,)
l2/W    : (100, 100)
---------------------------------------------

model.h5 saved!

--- The list of saved params in model.h5 ---
l1:
  W:(100, 784)
  b:(100,)
l2:
  W:(100, 100)
  b:(100,)
l3:
  W:(10, 100)
  b:(10,)
persistent: (10, 10)
---------------------------------------------
```

## Load

```
python load.py
```

It just loads `model.npz` to a model object (`chainer.Chain`) and also loads `model.h5` to another model object. Those model objects are created from a same class defined in `model.py` and should have the same parameters. So, this script also checks that all the loaded parameters are same.
The output will be as follows:

```bash
model.npz loaded!
model.h5 loaded!
/l3/W (10, 100)
/l3/b (10,)
/l1/W (100, 784)
/l1/b (100,)
/l2/W (100, 100)
/l2/b (100,)
```
