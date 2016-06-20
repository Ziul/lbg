from __future__ import print_function
from lbgargs import _parser
from glob import glob
import os
import json

from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import numpy as np
import itertools
import cv2
from codebooks import generate_codebook as compute
from mathematical import distortion, img_rate
from pkg_resources import resource_filename


_MAX_THREADS = 10
_pool = ThreadPool(processes=_MAX_THREADS)
(_options, _args) = _parser.parse_args()
levels = np.arange(10, 250, 30)


def axis_distortio(I1):
    R = []
    D = []
    U = []
    for lv in levels:
        I2 = I1.compress(tax=lv)
        U.append(len(np.unique(I2)))
        D.append(distortion(I1.figure, I2))
        R.append(img_rate(lv, I2.size / lv))
        I1.codebooks = []
    return D, R, U


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
    def v_centroids(data):
        """ Return the centroid of a conjunt of vectors."""
        return np.array(np.mean(np.mean(data, axis=0), axis=0))

    @staticmethod
    def avg(data):
        """ Return the avg of a conjunt of vectors."""
        return np.mean(data, axis=1)

    @staticmethod
    def norm(V1):
        """ Return the euclidian distance of two vectors n-size."""
        return np.linalg.norm(V1)

    @staticmethod
    def euclid(V1, V2):
        """ Return the euclidian distance of two vectors n-size."""
        return np.linalg.norm(V1, V2)

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        return np.array_split(l, n)

    @staticmethod
    def generate_data(i):
        global rgb_spectrum
        return np.array(compute(rgb_spectrum, i)[0])

    @staticmethod
    def external_fit(figure, codebooks):
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
        return LBG.external_fit(new_figure, codebooks)

    def fit(self, value):
        return LBG.external_fit(self.figure, self.codebooks)

    def compress(self, tax=10):
        if len(np.unique(self.figure)) <= tax:
            return self.figure
        if len(self.codebooks) == 0:
            self.generate_centroids(tax)
        return LBG.external_compress(self.figure, self.codebooks)

    @staticmethod
    def is_max_equal(old, new):
        try:
            if old.shape != new.shape:
                return False
        except Exception:
            return False

        return np.abs(new - old).max()

    @staticmethod
    def external_generate_centroids(values, tax=10):
        values = values.flatten()
        values.sort()
        unique_size = len(np.unique(values))
        if tax < 1:
            tax = np.ceil(tax * unique_size)

        values = np.array(np.array_split(values, tax))
        old = np.zeros(unique_size)
        centroids = np.array([np.mean(i) for i in values])
        while LBG.is_max_equal(old, centroids) > _options.error:
            old = centroids.copy()
            values = np.array_split(np.concatenate(
                values), np.mean(values, axis=1))
            centroids = np.array([np.mean(i) for i in values])
        codebooks = np.round(centroids)
        print('{} >> {} -> {}[{}]'.format(
            unique_size, tax, len(np.unique(centroids)),
            len(np.unique(codebooks))))

        return codebooks

    def generate_centroids(self, tax=10):
        values = np.concatenate(self.figure.copy())
        values.sort()
        unique_size = len(np.unique(values))
        if tax < 1:
            tax = np.ceil(tax * unique_size)

        values = np.array(np.array_split(values, tax))
        old = np.zeros(unique_size)
        centroids = np.array([np.mean(i) for i in values])
        while LBG.is_max_equal(old, centroids) > self.epsilon:
            old = centroids.copy()
            values = np.array_split(np.concatenate(
                values), np.mean(values, axis=1))
            centroids = np.array([np.mean(i) for i in values])
        # A lot of centroids vanishes here
        self.codebooks = np.round(centroids)
        print('{} >> {} -> {}[{}]'.format(
            unique_size, tax, len(np.unique(centroids)),
            len(np.unique(self.codebooks))))

        return self.codebooks


def show(figure, compressed_figure):
    figure.windows.add_subplot(2, 2, 1)
    plt.title('Original [M={}]'.format(len(np.unique(figure.figure))))
    plt.imshow(figure.figure, cmap='Greys_r')
    # plt.colorbar()

    figure.windows.add_subplot(2, 2, 3)
    plt.title('Histograma das imagens')
    plt.hist(compressed_figure.flatten(), bins=range(
        0, 256, 3), label="Quantizada")
    plt.hist(figure.figure.flatten(), bins=range(
        0, 256, 1), label='Original')
    if _options.log:
        plt .yscale('log')
    plt.legend()

    figure.windows.add_subplot(2, 2, 2)
    plt.title('Quantizada [M={}]'.format(len(np.unique(compressed_figure))))
    plt.imshow(compressed_figure, cmap='Greys_r')
    # plt.colorbar()

    if not _options.fast:
        figure.windows.add_subplot(2, 2, 4)
        plt.title('Histograma da quantizada')
        plt.title('Desigualdade x Taxa')
        D, R, U = axis_distortio(figure)
        plt.xlabel('Desigualdade')
        plt.ylabel('Taxa')
        plt.plot(D, R)
        for d, r, lv, u in zip(D, R, levels, U):
            plt.text(d, r, "{}[{}]".format(lv, u))
        if _options.save:
            name = _options.filename.split('.')
            name = name[:-1] + ['_compressed'] + name[-1:]
            cv2.imwrite('.'.join(name), compressed_figure)
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

    with plt.style.context(('seaborn-pastel')):
        # with plt.xkcd():
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
    codebook_filename = resource_filename(__name__, 'codebook.lbg')
    with open(codebook_filename, 'w')as txtfile:
        txtfile.write(json.dumps(list(codebooks), ensure_ascii=False))
    print(codebooks)


def apply_codebook():
    if (not _options.filename):
        if _args:
            _options.filename = _args[0]
        else:
            _parser.print_help()
            return
    if _options.compress != 10:
        print("Deprecated")

    codebook_filename = resource_filename(__name__, 'codebook.lbg')
    with open(codebook_filename, 'r')as txtfile:
        codebooks = json.load(txtfile)

    figure = LBG(_options.filename)
    compressed_figure = LBG.external_compress(figure.figure, codebooks)

    # with plt.xkcd():
    with plt.style.context(('seaborn-pastel')):
        show(figure, compressed_figure)


if __name__ == '__main__':
    main()
