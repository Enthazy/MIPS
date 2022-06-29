import matplotlib.pyplot as plt
import numpy as np
import sys


def load(file_name):
    """ Loads the model from numpy file.
    """
    print("Loading from " + file_name)
    return dict(np.load(file_name))


def display(qx, qy):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    N = len(qx) // 2
    # Now, we draw our points with a gradient of colors.
    ax.scatter(qx[:N], qy[:N], linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='teal', cmap=plt.cm.jet)
    ax.scatter(qx[N:], qy[N:], linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='coral', cmap=plt.cm.jet)

    ax.axis('equal')
    ax.grid()


def presenter(savepoint):
    data = load("./results/" + str(savepoint) + ".npz")
    qx = data['qx']
    qy = data['qy']
    print("=========Load savepoint successfully=========")
    display(qx, qy)
    plt.title(savepoint)


def generate_figure(filename):
    presenter(filename[:-4])
    plt.savefig("./fig/" + filename[:-3] + "png")
    plt.cla()
    plt.clf()
    plt.close('all')


if __name__ == "__main__":
    generate_figure(sys.argv[1])
