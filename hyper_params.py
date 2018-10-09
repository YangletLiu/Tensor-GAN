
class HyperParams():
    # tensor to block parameters
    patch_size = 3
    step = 1
    r = 150

    # TGAN parameters
    # gradient penalty coefficient
    beta = 10
    # content loss coefficient
    sigma1 = 0.001
    # discriminator loss coefficient
    sigma2 = 1
    # generator loss coefficient
    sigma3 = 1

    z_dim = 32
    learning_rate = 1e-4
    batch_size = 32
