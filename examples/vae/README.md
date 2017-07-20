# Variational AutoEncoder

This is a sample implementation of variational autoencoder.

This method is proposed by Kingma and Welling, Auto-Encoding Variational Bayes, 2014.

If you want to run this example on the N-th GPU, pass `--gpu=N` to the script.

If you run this script on Linux, setting the environmental variable `MPLBACKEND` to `Agg` may be required to use `matplotlib`. For example,

```
MPLBACKEND=Agg python train_vae.py
```
