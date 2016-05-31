from lbgargs import _parser
from pprint import pprint
from sys import argv, stdout

from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import itertools
import cv2
from codebooks import generate_codebook as compute
from codebooks import avg_distortion_c0 as avg_distortion

_MAX_THREADS = 10
_pool = ThreadPool(processes=_MAX_THREADS)
(_options, _args) = _parser.parse_args()

import concurrent.futures


def print(*data, end='\n'):
    # overwrite print
    if _options.verbose:
        for i in data:
            stdout.write(str(i) + ' ')
        stdout.write(end)


class LBG(object):
    """docstring for LBG"""
    epsilon = 1e-3

    def __init__(self, filename):
        super(LBG, self).__init__()
        self.filename = filename
        # self.figure = plt.imread(filename)
        self.figure = cv2.imread(filename)
        self.windows = plt.figure()

    @staticmethod
    def combinations(data):
        """ return all the possible combinations of a list of lists."""
        return list(itertools.product(*data))

    @staticmethod
    def centroids(data):
        """ Return the centroid of a conjunt of vectors."""
        out = []
        for i in data:
            out.append(i.mean(axis=0))
        return np.array(out)

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

    # @staticmethod
    def convulate(self, value):
        codebooks = self.codebooks
        for i in range(len(codebooks) - 1):
            if codebooks[i] <= value <= codebooks[i + 1]:
                if (value - codebooks[i]) < (codebooks[i + 1] - value):
                    return codebooks[i]
                else:
                    return codebooks[i + 1]
        if value < codebooks.min():
            return codebooks.min()
        else:
            return codebooks.max()

    def is_avg_equal(self, old, new):
        if (old == new).all():
            print('CRITICO!')
            return 0
        print(np.abs(new - old).max())
        return np.abs(new - old).max()

    def compress(self, tax):
        self.windows.add_subplot(2, 2, 1)
        plt.imshow(self.figure.copy())
        self.windows.add_subplot(2, 2, 3)
        plt.hist(self.figure[0])
        print(self.figure.shape)
        red = (self.figure[::, 0]).flatten()
        green = (self.figure[::, 1]).flatten()
        blue = (self.figure[::, 2]).flatten()
        if tax < 1:
            # tax = np.ceil(tax * len(np.unique(red)))
            tax = np.ceil(tax * len(red))

        old = np.array([0])
        new = np.array([1])
        # while self.is_avg_equal(old, new) > self.epsilon:
        old = new.copy()
        new_red = np.array_split(red, tax)
        self.codebooks = LBG.centroids(new_red)
        new = self.codebooks.copy()
        red = np.array([])
        for reds in new_red:
            red = np.append(red, _pool.map(self.convulate, reds))
        red = red.flatten()
        self.windows.add_subplot(1, 2, 2)
        plt.hist(red)

        print(len(red))
        red.shape = (self.figure.shape[0], 3)
        green.shape = (self.figure.shape[0], 3)
        blue.shape = (self.figure.shape[0], 3)
        # self.figure[::, 0] = red
        # print(self.figure)
        with plt.xkcd():

            # plt.imshow(self.figure)
            # cv2.imshow('olar', np.array(
            #     [red, green, blue], dtype=self.figure.dtype).T)
            # plt.show()
            plt.show()
        # print(LBG.avg(new_red))
        # print(list(zip(codebooks, new_red)))

    def compress_by_reduction(self, tax):
        red = self.figure[:, 0] * 255
        green = self.figure[:, 1] * 255
        blue = self.figure[:, 2] * 255
        if tax < 1:
            tax = 1 / tax

        k = np.unique(np.round(red))
        print(k, '-', len(k))
        red = compute(red, tax)
        green = compute(green, tax)[0]
        blue = compute(blue, tax)[0]

        print(red[0])
        # k = np.unique(np.round(red))
        # print(k, '-', len(k))

        # return np.array([red, green, blue], dtype=self.figure.dtype).T


def main():
    if (not _options.filename) and (not _options.text):
        if _args:
            _options.filename = _args[0]
        else:
            _parser.print_help()
            return
    if not _options.compress:
        raise ValueError('Compress tax not informed')

    figure = LBG(_options.filename)
    figure.compress(_options.compress)


def test():
    global rgb_spectrum

    def update_plot(i, data, scat):
        from mpl_toolkits.mplot3d.art3d import juggle_axes
        paleta = list(zip(*data[i]))
        scat._offsets3d = juggle_axes(paleta[0], paleta[1], paleta[2], 'z')
        return scat,
    rgb = np.array([np.arange(1, 256.0, 25), np.arange(
        1, 256.0, 25), np.arange(1, 256.0, 25)])
    # rgb = np.array([np.arange(4), np.arange(3, 7), np.arange(10, 14)])
    rgb_spectrum = LBG.combinations(rgb)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.pyplot import cm
    import random

    # max_size = 65
    min_size = 4
    max_size = 7
    # spectrum = np.array(other.generate_codebook(rgb_spectrum, count)[0])
    spectrum = _pool.map(
        LBG.generate_data, [2**j for j in range(min_size, max_size + 1)])
    # size_codebook = 17
    # rgb_spectrum = other.generate_codebook(rgb_spectrum, size_codebook)[0]

    with plt.xkcd():
        paleta = next(zip(*spectrum[-1]))
        print(rgb_spectrum[0], rgb_spectrum[-1])
        fig = plt.figure()
        ax = Axes3D(fig)
        scat = ax.scatter(paleta[0], paleta[1], paleta[2],
                          s=200, depthshade=False, marker='s',
                          c=[(r[0] / 256., r[1] / 256., r[2] / 256.) for r in spectrum[-1]])
        # ax.grid(False)
        ax.set_title('grid on')

        plt.title('RGB Spectrum')

        ani = animation.FuncAnimation(fig, update_plot, frames=range(max_size - min_size + 1),
                                      fargs=(spectrum, scat))

        FFwriter = animation.FFMpegWriter()
        # ani.save('rgb_spectrum.mp4', writer=FFwriter, fps=30)

        plt.show(fig)                     # display the plot

        # plt.show()

if __name__ == '__main__':
    main()
