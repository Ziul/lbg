import numpy as np


def distortion(I1, I2):
    """ Distortion of two images """
    return ((I1 - I2)**2).mean()


def img_rate(M, N):
    """
    Return the img_rate of the quantization

    @param M: levels of quantization
    @param N: number of pixels
    """
    return np.log2(M) / N
