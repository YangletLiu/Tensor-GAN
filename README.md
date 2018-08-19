# Tensor-GAN

## Tensor Sparse Coding

### Numpy
- Run `main.py` to get sparse representation of a 3D tensor. 
- The default sample is in the folder `./samples`, which is a `101*101*31` pixel picture.
- The result of default sample is as follow, where the parameters are set in file `hyper_params.py`:
 ![image](https://github.com/hust512/Tensor-GAN/blob/master/baloon_sc_result.png)
 
 ### Tensorflow
- The version of Tensorflow has been implemented, whose codes are in the folder `./tensorflow`. Please run `train.py` to start. Then the reconstructions by tensor sparses will be saved in `./tensorflow/out/`.
- The result of Tensorflow is also as follow with same parameters:
![image](https://github.com/hust512/Tensor-GAN/blob/master/tensorflow/baloon_sc_tensorflow.png)
