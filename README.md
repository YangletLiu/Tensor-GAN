# Tensor Sparse Coding

- [1] F. Jiang, X.-Y. Liu, H. Lu, R. Shen. Efficient multi-dimensional tensor sparse coding using t-linear combinations. AAAI, 2018.

- The version of Python is in the repository `./2DSC`. Please run `main.py` to start. And the version of Tensorflow is in the repository `./2DSC/tensorflow`. Please run `train.py` to start. 

![image](https://github.com/hust512/Tensor-GAN/blob/master/pics/balloon_sc_result.png)

<div align=center>Figure 1. The experiment of tensor sparse coding.</div>

# Tensor GAN

## Introduction
- The codes are in the repository `./TGAN`.

## Usage
Basicly, we select MNIST dataset as an example.
- Firstly, run `GAN.py` to train a generative model which is used to generate low-quality tensors with size `14 * 14 * 7` from a random distribution. 
- Secondly, run `gen_training_data.py`, which is used to generate training data consists of pairs of low-quality images and high-qulity images. As downscaled from the original MNIST images with size `28 * 28`, the low-quality image is of size `14 * 14 * 7`, which is concatenated by single images shifted with different pixels. 
- Then, run `./SR/D_Training.m` to train a combined tensor dictionary.
- Lastly, run `gen_test_data.py` to use GANs to generate low-quality images. And run `./SR/Demo_SR.m` to generate high-qulity images.

## Dataflow
- Dataset: In order to adapt the data to our tensor theory, the MNIST dataset are transformed to the image tensors (HT) with size `28 * 28 * 7`, concatenating `7` images which are shifted from the same single image with different pixels. Then downscaling the HT with `2x`, we get low-resolution image tensors (LT), which is of size `14 * 14 * 7`.

- Low-resolution images generating: A DCGAN with gradient penalty is applied to generate low-resolution images (LS) with size `14 * 14 * 1` from a latent vector with size `128 * 1`. Similarly, we concatenate `7` shifted LIs into tensors with size `14 * 14 * 7`, named LTS.

- Tensor dictionary training: HT and LT are used to train a high resolution dictionary (HD) and a low resolution dictionary (LD) which have a same shape. With the same sparse coefficients caculated by the LD, a high-resolution image tensor (HTS) with shape `28 * 28 * 7` is generated from the low-resolution image tensor sample (LTS). 

- Dimensional variation: latent vector (`128 * 1`) →(DCGAN) LS (`14 * 14 * 1`) → LTS (`14 * 14 * 7`) →(SR) HTS (`28 * 28 * 7`)

## Architechture
<div align=center><img width="800" src="https://github.com/hust512/Tensor-GAN/blob/master/pics/arch.jpg"/></div>

<div align=center>Figure 2. The architechture of tensor GAN.</div>

<div align=center><img width="600" src="https://github.com/hust512/Tensor-GAN/blob/master/pics/dict.png"/></div>
<div align=center> Figure 3. The tensor dictionary.</div>

## Experiment
We have tested the high-quality image generation from a low-resolution image via tensor combined dictionary.

<div align=center><img width="400" src="https://github.com/hust512/Tensor-GAN/blob/master/pics/balloons_sr_result.png"/></div>

And the following pictures are the generated handwritten numbers the model. The above pictures are generated from latent representations by a generator, which is composed of an adversarial autoencoder networks. The pictures at the bottom are generated by our tensor superresolution (essentially tensor sparse coding) from the above pictures.

<div align=center><img width="500" src="https://github.com/hust512/Tensor-GAN/blob/master/pics/mnist.png"/></div>
