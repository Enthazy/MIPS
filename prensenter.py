import matplotlib.pyplot as plt
import sys
from time import time
import numpy as np

class Presenter:
    def __init__(self):
        pass

    def plot(self, x, y, r=3,**kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # Now, we draw our points with a gradient of colors.
        ax.scatter(x, y, linewidths=0,
                   marker='o', s=10, cmap=plt.cm.jet)
        ax.axis('equal')
        # ax.set_axis_off()
        plt.show()
def save(file_name, data):
    """ Saves the model to a numpy file.
    """
    print("Writing to " + file_name)
    np.savez_compressed(file_name, **data)


def load(file_name):
    """ Loads the model from numpy file.
    """
    print("Loading from " + file_name)
    return dict(np.load(file_name))

def display(posx, posy):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Now, we draw our points with a gradient of colors.
    ax.scatter(np.remainder(posx[:len(posx)//2],1), np.remainder(posy[:len(posx)//2],1), linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='teal', cmap=plt.cm.jet)
    ax.scatter(np.remainder(posx[len(posx)//2:],1), np.remainder(posy[len(posx)//2:],1), linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='coral', cmap=plt.cm.jet)

    ax.axis('equal')
    ax.grid()
    plt.savefig("./results/view.png")

if __name__=="__main__":
    file_name = "./results/state4000.npz"
    data = load(file_name)
    display(data['px'],data['py'])
    print(data['px'])