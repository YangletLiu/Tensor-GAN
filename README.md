# Tensor-GAN

## Tensor Sparse Coding

### Numpy
- Run `main.py` to get sparse representation of a 3D tensor. 
- The default sample is in the folder `./samples`, which is a `101*101*31` pixel picture.
- The result of default sample is as follows, where the parameters are set in file `hyper_params.py`:
 ![image](https://github.com/hust512/Tensor-GAN/blob/master/balloon_sc_result.png)
 
 ### Tensorflow
- The version of Tensorflow has been implemented, whose codes are in the folder `./tensorflow`. Please run `train.py` to start. Then the reconstructions by tensor sparses will be saved in `./tensorflow/out/`.
- The result of Tensorflow is also as follows with same parameters:
![image](https://github.com/hust512/Tensor-GAN/blob/master/tensorflow/balloon_sc_tensorflow.png)

## Tensor SC with GAN
- A sample result:
![image](https://github.com/hust512/Tensor-GAN/blob/master/tensorflow/tgan_sample.png)
