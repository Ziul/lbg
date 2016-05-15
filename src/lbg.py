from lbgargs import _parser
from pprint import pprint

from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import itertools

_MAX_THREADS = 10
_pool = ThreadPool(processes=_MAX_THREADS)
(_options, _args) = _parser.parse_args()


def centroid(data):
    """ Return the centroid of a conjunt of vectors."""
    return data.mean(axis=0)


def mean(data):
    """ Return the mean of each conjunt of vectors."""
    return data.mean(axis=1)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    data = []
    for i in range(0, len(l), n):
        # yield l[i:i + n]
        data.append(l[i:i + n])

    return data


def combinations(data):
    """ return all the possible combinations of a list of lists."""
    return list(itertools.product(*data))


class LBG(object):
    """docstring for LBG"""

    def __init__(self, filename):
        super(LBG, self).__init__()
        self.filename = filename


def main():
    if (not _options.filename) and (not _options.text):
        if _args:
            _options.filename = _args[0]
        else:
            _parser.print_help()
            return


def generate_data(i):
    import other
    global rgb_spectrum
    return np.array(other.generate_codebook(rgb_spectrum, i)[0])


def update_plot(i, data, scat):
    from mpl_toolkits.mplot3d.art3d import juggle_axes
    paleta = zip(*data[i])
    scat._offsets3d = juggle_axes(paleta[0], paleta[1], paleta[2], 'z')
    return scat,


def test():
    global rgb_spectrum
    rgb = np.array([np.arange(1, 256.0, 25), np.arange(
        1, 256.0, 25), np.arange(1, 256.0, 25)])
    # rgb = np.array([np.arange(4), np.arange(3, 7), np.arange(10, 14)])
    rgb_spectrum = combinations(rgb)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.pyplot import cm
    import random

    # max_size = 65
    min_size = 4
    max_size = 7
    # spectrum = np.array(other.generate_codebook(rgb_spectrum, count)[0])
    spectrum = _pool.map(
        generate_data, [2**j for j in range(min_size, max_size + 1)])
    # size_codebook = 17
    # rgb_spectrum = other.generate_codebook(rgb_spectrum, size_codebook)[0]

    with plt.xkcd():
        paleta = zip(*spectrum[-1])
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
