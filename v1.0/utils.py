import numpy as np
import matplotlib.pyplot as plt
import math
from time import time

from numba import jit, njit, prange
from numba.typed import List
from matplotlib.animation import FuncAnimation, PillowWriter


@njit()
def set_seed(a):
    np.random.seed(a)

def display(posx, posy):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Now, we draw our points with a gradient of colors.
    ax.scatter(posx[:len(posx)//2], posy[:len(posx)//2], linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='teal', cmap=plt.cm.jet)
    ax.scatter(posx[len(posx)//2:], posy[len(posx)//2:], linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='coral', cmap=plt.cm.jet)

    ax.axis('equal')
    ax.grid()
    # ax.set_axis_off()


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

def cal_folding(N, Lx, Ly):
    return np.pi * N / (Lx * Ly)

def cal_Pe(sigma, v, D):
    '''
    Calculate the Peclet number
    '''
    t = sigma**2 / D
    Pe = t * v /sigma
    return Pe

def record_gif(data):
    fig,ax = plt.subplots()
    def animate(i):
        fig.clear()
        line = display(data[i][0],data[i][1])
        ax.set_title("iteration " + str(i))
        return line

    ani = FuncAnimation(fig, animate, interval=200, frames=354)

    ani.save("test0.gif", dpi=200, writer=PillowWriter(fps=24))

def generate_points_with_min_distance(n, shape, min_dist):
    """
        n: number of points
        example:
        generate_points_with_min_distance(n=100, shape=(100,100), min_dist=0.1)
    """
    # compute grid shape based on number of points
    width_ratio = shape[1] / shape[0]
    num_y = np.int32(np.sqrt(n / width_ratio))
    num_x = np.int32(n / num_y)
    # create regularly spaced neurons
    x = np.linspace(0., shape[1], num_x, dtype=np.float32)
    y = np.linspace(0., shape[0], num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)

    # compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))

    # perturb points
    max_movement = (init_dist - min_dist)/2
    noise = np.random.uniform(low=-max_movement,
                              high=max_movement,
                              size=(len(coords), 2))
    coords += noise

    return coords