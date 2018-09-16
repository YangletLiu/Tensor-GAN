
class HyperParams():
    # tensor to block parameters
    patch_size = 3
    step = 1
    r = 20

    # TGAN parameters
    # gradient penalty coefficient
    beta = 10
    # content loss coefficient
    sigma1 = 1
    # discriminator loss coefficient
    sigma2 = 1e-2
    # generator loss coefficient
    sigma3 = 1e-2

    z_dim = 128
    learning_rate = 0.01
    batch_size = 32
