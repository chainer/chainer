# chainer-pix2pix
chainer implementation of pix2pix
https://phillipi.github.io/pix2pix/

The Japanese readme can be found [here](README-ja.md).

# Example result on CMP facade dataset
<img src="https://github.com/mattya/chainer-pix2pix/blob/master/image/example.png?raw=true">
From the left side: input, output, ground_truth


# usage
1. `pip install -r requirements.txt`
2. Download the facade dataset (base set) http://cmp.felk.cvut.cz/~tylecr1/facade/
 - `wget http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip`
 - `unzip CMP_facade_DB_base.zip`
3. `python train_facade.py -g [GPU ID, e.g. 0] -i [dataset root directory] --out [output directory] --snapshot_interval 10000`
4. Wait a few hours...
 - `--out` stores snapshots of the model and example images at an interval defined by `--snapshot_interval`
 - If the model size is large, you can reduce `--snapshot_interval` to save resources.

# Using other datasets
- Gather image pairs (e.g. label + photo). Several hundred pairs are required for good results.
- Create a copy of `facade_dataset.py` for your dataset. The function get_example should be written so that it returns the i-th image pair a tuple of numpy arrays i.e. `(input, output)`.
- It may be necessary to update the loss function in `updater.py`.
- Likewise, make a copy of `facade_visualizer.py` and modify to visualize the dataset.
- In `train_facade.py` change `in_ch` and `out_ch` to the correct input and output channels for your data.
