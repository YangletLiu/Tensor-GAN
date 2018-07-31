import tensorly
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import skimage.data
import skimage.color

img = skimage.transform.rescale(skimage.data.chelsea(), 0.2)
img = skimage.color.rgb2gray(img)



