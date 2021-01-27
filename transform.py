import random
import numpy as np
import torch

# Add graussian noise with zero mean and standard deviation 0.01 to 0.04
class AddGaussianNoise(object):
    def __call__(self, x):
        var = random.random() * 0.04 + 0.01
        noise = np.random.normal(0, var, (1000))
        x += noise
        x = np.clip(x, 0, 1)
        return x


# Average blur with widow size 1 to 5
class RandomBlur(object):
    def __call__(self, x):
        size = random.randint(1, 5)
        x = np.convolve(x, np.ones(size) / size, mode='same')
        return x


# randomly set the intensity of spectrum to 0
class RandomDropout(object):
    def __call__(self, x, droprate=0.1):
        noise = np.random.random(1000)
        x = (noise > droprate) * x
        return x

# mulitiply the spectrum by a scale-factor
class RandomScaleTransform(object):
    def __call__(self, x):
        scale = np.random.uniform(0.9, 1.1, x.shape)
        x = scale * x
        x = np.clip(x, 0, 1)
        return x


# convert to Tensor with 1 channel
class ToFloatTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x).view(1, -1).float()
