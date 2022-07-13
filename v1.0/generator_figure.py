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
    import os
    os.makedirs("./fig/", exist_ok=True)
    plt.savefig("./fig/" + filename[:-3] + "png")
    plt.cla()
    plt.clf()
    plt.close('all')

def generate_LJ():
    domain = np.arange(0.93, 2.6, 0.001)
    s = 4*(domain**(-12)-domain**(-6))
    plt.plot(domain,s, label="L-J potential", color='teal',linewidth=2)
    s2 = [x+1 if i <190 else 0 for i,x in enumerate(s)]
    plt.plot(domain,s2, label='modified L-J potential', color='coral', linewidth=2)
    plt.grid(alpha=0.5)
    plt.xlabel("r/$\sigma$")
    plt.ylabel("$V_{LJ}/\epsilon$")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    generate_figure(sys.argv[1])
    # generate_LJ()