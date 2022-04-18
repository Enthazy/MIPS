import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


@njit()
def seed(a):
    np.random.seed(a)


@njit()
def init(N):
    """
    posx, posy: (N,) ndarray, position
    theta: (N,) ndarray, [0,1] represent [0, 2pi], the angle that the particle toward
    """
    posx = np.random.uniform(0, 1, N)
    posy = np.random.uniform(0, 1, N)
    theta = np.random.randn(N)

    vecx = np.zeros(N)
    vecy = np.zeros(N)
    return posx, posy, vecx, vecy, theta


@njit()
def pos_updater(posx, posy, vecx, vecy, theta,
                N, step, v_p, D_r, D_t, gamma):

    # Interaction and Self-drive Motion
    posx += step * (vecx + v_p * np.cos(theta)) / gamma
    posy += step * (vecy + v_p * np.sin(theta)) / gamma

    # Diffusion
    posx += np.sqrt(2*D_t*step)*np.random.randn(N)
    posy += np.sqrt(2*D_t*step)*np.random.randn(N)
    theta += np.sqrt(2*D_r*step)*np.random.randn(N)


@njit()
def LJPotential(x, y, d2, r2, epsilon):
    '''
        calculat the FJ potential
    :param x: difference on x
    :param y: difference on y
    :param d: distance sqrt(x^2 + y^2)
    :param r6: 6 power of radius of particles r , r^6
    :param epsilon: the potential strength
    :return:
    '''
    rx = x/d2
    ry = y/d2
    k = (r2/d2)**3
    s = epsilon * (k - 2 * k**2)
    return s*rx, s*ry


@njit(parallel=True)
def vec_updater(posx, posy, vecx, vecy, theta,
                N, r, r2, epsilon):

    r_lower = -1.5 * r
    r_upper = 1.5 * r

    # Run with numba parallel iteration
    for i in prange(N):

        #======================================================
        # Collide Detection
        index_range = range(i + 1, N)  # check all particles

        for j in index_range:

            # check if two particles are contacted, most particle are not contacted
            x1 = posx[i] % 1
            x2 = posx[j] % 1
            x_diff = x1 - x2

            if r_lower < x_diff < r_upper:

                y1 = posy[i] % 1
                y2 = posy[j] % 1
                y_diff = y1 - y2

                if r_lower < y_diff < r_upper:

                    d2 = x_diff ** 2 + y_diff ** 2 + 1e-16
                    if d2 < r2:
                        # if contact add the repulsive force
                        vpx, vpy = LJPotential(x_diff, y_diff, d2, r2, epsilon)
                        vecx[i] += vpx
                        vecy[i] += vpy
                        vecx[j] -= vpx
                        vecy[j] -= vpy


@njit()
def run(posx, posy, vecx, vecy, theta,
        N, r, r2, epsilon, step, v_p, D_r, D_t, gamma):
    vec_updater(posx, posy, vecx, vecy, theta, N, r, r2, epsilon)
    pos_updater(posx, posy, vecx, vecy, theta, N, step, v_p, D_r, D_t, gamma)
    vecx = np.zeros(N)
    vecy = np.zeros(N)



def display(posx, posy):

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Now, we draw our points with a gradient of colors.
    ax.scatter(np.remainder(posx,1), np.remainder(posy,1), linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='teal', cmap=plt.cm.jet)
    ax.axis('equal')
    ax.grid()
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


if __name__ == '__main__':
    # Hyper-parameters
    N = 20000  # number of particles
    step = 3e-2
    epoch = 2
    r = 1e-3
    r2 = r ** 2
    r6 = r ** 6  # radius of particle power 6
    gamma = 1  # fraction
    epsilon = 1e-10  # potential depth times 4
    v_p = 0.25  # mean speed
    D_t = 1e-3  # transition diffusion
    D_r = 1e-3  # rotation diffusion
    is_save = False
    is_load = False

    # Initialization
    posx, posy, vecx, vecy, theta = init(N)
    display(posx, posy)

    # Run
    for _ in range(epoch):
        print("iteration: ", _)
        run(posx, posy, vecx, vecy, theta,
            N, r, r2, epsilon, step, v_p, D_r, D_t, gamma)

    if is_save:
        data = {'px': posx,
                'py': posy,
                'vx': vecx,
                'vy': vecy,
                'theta': theta
                }
        save("state.npz", data)
    display(posx, posy)
