# chainer-pix2pix
chainer implementation of pix2pix
https://phillipi.github.io/pix2pix/

# Example result on CMP facade dataset
<img src="https://github.com/mattya/chainer-pix2pix/blob/master/image/example.png?raw=true">
input, output, ground_truth


# usage
1. install chainer and cupy
2. download facade dataset
 - `wget http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip`
 - `unzip CMP_facade_DB_base.zip`
3. `python train_facade.py -g [GPU ] -i [dataset root directory] --out [result directory] --snapshot_interval 10000`
4. training would be take several ours
 - snapshot image will be saved in directory at `--out`, with an interval set by `--snapshot_interval`

