from __future__ import print_function
from lbgargs import _parser
from pprint import pprint
from sys import argv, stdout
from glob import glob
import os
import sys
import fnmatch
import json

from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import itertools
import cv2
from codebooks import generate_codebook as compute
from codebooks import avg_distortion_c0 as avg_distortion
from mathematical import distortion, img_rate

_MAX_THREADS = 10
_pool = ThreadPool(processes=_MAX_THREADS)
(_options, _args) = _parser.parse_args()


def print(*args, end='\n'):
    if not _options.verbose:
        return
    for c in args:
        stdout.write(str(c))
    stdout.write(end)


def axis_distortio(I1):
    levels = [5, 10, 30, 50, 70, 100, 120, 150, 200, 250]
    R = []
    D = []
    for lv in levels:
        I2 = I1.compress(tax=lv)
        D.append(distortion(I1.figure, I2))
        R.append(img_rate(lv, I2.size / lv))
        I1.codebooks = []
    return D, R


class FileNotFound(Exception):
    """docstring for FileNotFound"""

    def __init__(self, arg=''):
        super(FileNotFound, self).__init__('Figure not found: ' + arg)


class LBG(object):
    """docstring for LBG"""
    epsilon = _options.error

    def __init__(self, filename):
        super(LBG, self).__init__()
        self.filename = filename
        # self.figure = plt.imread(filename)
        self.figure = cv2.imread(filename, 0)
        if self.figure is None:
            raise FileNotFound(filename)
        self.windows = plt.figure()
        self.codebooks = []

    @staticmethod
    def combinations(data):
        """ return all the possible combinations of a list of lists."""
        return list(itertools.product(*data))

    @staticmethod
    def centroids(data):
        """ Return the centroid of a conjunt of vectors."""
        return np.array([np.mean(i) for i in data])

    @staticmethod
    def avg(data):
        """ Return the avg of a conjunt of vectors."""
        return np.mean(data, axis=1)

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        return np.array_split(l, n)

    @staticmethod
    def generate_data(i):
        global rgb_spectrum
        return np.array(compute(rgb_spectrum, i)[0])

    @staticmethod
    def external_convulate(figure, codebooks):
        for i in range(len(codebooks) - 1):
            mask = np.ma.masked_inside(figure, codebooks[i], codebooks[i + 1])
            figure[mask.mask] = codebooks[i]

        min = np.min(codebooks)
        max = np.max(codebooks)
        figure[figure <= min] = min
        figure[figure >= max] = max
        return figure

    @staticmethod
    def external_compress(figure, codebooks):
        new_figure = figure.copy()
        return LBG.external_convulate(new_figure, codebooks)

    def convulate(self, value):
        return LBG.external_convulate(self.figure, self.codebooks)

    def compress(self, tax=10):
        if len(self.codebooks) == 0:
            self.generate_centroids(tax)
        return LBG.external_compress(self.figure, self.codebooks)

    def is_max_equal(self, old, new):
        try:
            if old.shape != new.shape:
                return False
        except Exception:
            return False

        return np.abs(new - old).max()

    def generate_centroids(self, tax=10):
        values = np.concatenate(self.figure.copy())
        values.sort()
        unique_size = len(np.unique(values))
        if tax < 1:
            tax = np.ceil(tax * unique_size)

        values = np.array(np.array_split(values, tax))
        old = np.zeros(unique_size)
        centroids = np.array([np.mean(i) for i in values])
        while self.is_max_equal(old, centroids) > self.epsilon:
            old = centroids.copy()
            values = np.array_split(np.concatenate(
                values), np.mean(values, axis=1))
            centroids = np.array([np.mean(i) for i in values])
        self.codebooks = np.round(centroids)
        print('{} >> {}'.format(unique_size, len(centroids)))

        return self.codebooks


def show(figure, compressed_figure):
    figure.windows.add_subplot(2, 2, 1)
    plt.title('Original')
    plt.imshow(figure.figure, cmap='Greys_r')
    # plt.colorbar()
    figure.windows.add_subplot(2, 2, 3)
    plt.title('Histograma das imagens')
    plt.hist(figure.figure.flatten(), bins=range(
        0, 256, 5), label='Original')
    plt.hist(compressed_figure.flatten(), bins=range(
        0, 256, 5), label="Quantizada")
    # plt.hist([figure.figure.flatten(), compressed_figure.flatten()],
    #          bins=range(0, 255, 5))
    plt.legend()
    figure.windows.add_subplot(2, 2, 2)
    plt.title('Quantizada [M={}]'.format(len(np.unique(compressed_figure))))
    plt.imshow(compressed_figure, cmap='Greys_r')
    # plt.colorbar()
    figure.windows.add_subplot(2, 2, 4)
    # plt.title('Histograma da quantizada')
    plt.title('Distorção x Taxa')
    D, R = axis_distortio(figure)
    plt.xlabel('Distorção')
    plt.ylabel('Taxa')
    plt.plot(D, R)
    plt.show()


def main():
    if (not _options.filename):
        if _args:
            _options.filename = _args[0]
        else:
            _parser.print_help()
            return
    if not _options.compress:
        raise ValueError('Compress tax not informed')

    figure = LBG(_options.filename)
    compressed_figure = figure.compress(_options.compress)

    # with plt.style.context(('seaborn-pastel')):
    with plt.xkcd():
        show(figure, compressed_figure)


def learn():
    if (not _options.filename):
        if _args:
            _options.filename = _args[0]
        else:
            _parser.print_help()
            return
    if not os.path.isdir(_options.filename):
        raise Exception('Should pass a path with figures')
    try:
        LBG(_options.filename)
    except FileNotFound:
        print('OK')
    files = []
    for extencion in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
        files += glob(_options.filename + '/**/' + extencion, recursive=True)
    codebooks = np.ones(_options.compress)

    for f in files:
        print('\r{:.2f}% [{}] '.format(
            100. * (files.index(f) / len(files)), f), end='')
        try:
            figure = LBG(f)
            plt.close()
            try:
                figure.compress(_options.compress)
                codebooks = (codebooks + figure.codebooks) / 2
            except ValueError:
                pass
        except FileNotFound:
            pass
    codebooks = np.round(codebooks)
    print('\r100.00% with {} images'.format(len(files)))
    plt.cla()
    with open('codebook.lbg', 'w')as txtfile:
        txtfile.write(json.dumps(list(codebooks), ensure_ascii=False))
    print(codebooks)


def apply_codebook():
    if (not _options.filename):
        if _args:
            _options.filename = _args[0]
        else:
            _parser.print_help()
            return
    with open('codebook.lbg', 'r')as txtfile:
        codebooks = json.load(txtfile)

    figure = LBG(_options.filename)
    compressed_figure = LBG.external_compress(figure.figure, codebooks)

    # with plt.xkcd():
    with plt.style.context(('seaborn-pastel')):
        show(figure, compressed_figure)


if __name__ == '__main__':
    main()
