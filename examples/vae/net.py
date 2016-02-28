import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.vae import gaussian_kl_divergence
from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy


class VAE(chainer.Chain):
    """
    Variational AutoEncoder
    """
    def __init__(self, n_in, n_latent, n_h):
        super(VAE, self).__init__(
            # encoder 
            le1=L.Linear(n_in, n_h),
            le2_mu=L.Linear(n_h, n_latent),
            le2_ln_var=L.Linear(n_h, n_latent),
            # decoder
            ld1=L.Linear(n_latent, n_h),
            ld2=L.Linear(n_h, n_in),
        )
        self.n_latent = n_latent

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1) # log(sigma**2)
        return mu, ln_var
        
    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))        
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def get_loss_func(self, C=1.0, L=1, train=True):
        """
        Args:
            C (int): 1.0
            L (int): Number of Monte Carlo samples used in encoded vector.
            train (bool): if true loss_function is used for training.
        """
        def lf(x):
            _, n_in = x.data.shape
            mu, ln_var = self.encode(x)
            batchsize, _ = mu.data.shape
            # binarize input
            xi = chainer.variable.Variable((x.data > self.xp.random.random(x.data.shape)).astype(self.xp.int32), volatile=not train)
            # reconstruction loss
            rec_loss = 0
            for l in range(L):
                e = self.xp.random.normal(size=(batchsize, self.n_latent))        
                z = mu + F.exp(0.5 * ln_var) * e
                rec_loss += (1.0/L) * sigmoid_cross_entropy(self.decode(z, sigmoid=False), xi) 

            self.rec_loss = rec_loss * n_in
            self.loss = self.rec_loss + C * gaussian_kl_divergence(mu, ln_var) / batchsize           
            return self.loss
        return lf
        
        
        
        

        
        

    
