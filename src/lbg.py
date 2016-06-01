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


class LBG(object):
    """docstring for LBG"""
    epsilon = 1e-3

    def __init__(self, filename):
        super(LBG, self).__init__()
        self.filename = filename
        # self.figure = plt.imread(filename)
        self.figure = cv2.imread(filename, 0)
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
        try:
            if old.shape != new.shape:
                return False
        except Exception:
            return False
        if (old == new).all():
            print('CRITICO!')
            return 0
        print(np.abs(new - old).max())
        return np.abs(new - old).max()

    def compress(self, tax):
        self.windows.add_subplot(2, 2, 1)
        plt.imshow(self.figure.copy(), cmap='Greys_r')
        values = np.concatenate(self.figure.copy())
        values.sort()
        unique_size = len(np.unique(values))
        if tax < 1:
            tax = np.ceil(tax * unique_size)
        print('{} >> {}'.format(unique_size, tax))

        values = np.split(values, tax)
        old = np.zeros(unique_size)
        centroids = np.mean(values, axis=1)
        while self.is_avg_equal(old, values) > self.epsilon:
            old = centroids.copy()
            values = np.split(np.concatenate(values), np.mean(values, axis=1))
            centroids = LBG.avg(values)
        self.codebooks = centroids
        print '{} >> {}'.format(unique_size, len(centroids))

        with plt.xkcd():
            self.windows.add_subplot(2, 2, 3)
            plt.hist(self.figure[0], bins=range(255))
            self.windows.add_subplot(1, 2, 2)
            plt.hist(centroids, bins=range(255))
            plt.show()

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
