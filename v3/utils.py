import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit
# import seaborn as sns
import pandas as pd


@njit()
def set_seed(a):
    np.random.seed(a)


def display(posx, posy):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Now, we draw our points with a gradient of colors.
    ax.scatter(posx[:len(posx) // 2], posy[:len(posx) // 2], linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='teal', cmap=plt.cm.jet)
    ax.scatter(posx[len(posx) // 2:], posy[len(posx) // 2:], linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='coral', cmap=plt.cm.jet)

    ax.axis('equal')
    ax.grid()
    # ax.set_axis_off()

# def sns_display(posx, posy):
#     data = pd.DataFrame({'px':posx, 'py':posy})
#     sns.scatterplot(x='px',y='py',data=data)
#     plt.grid(alpha=0.5)


def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection

def display_ball(qx, qy):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # Now, we draw our points with a gradient of colors.
    circles(qx[:len(qx) // 2], qy[:len(qx) // 2], 1 / 2, c='teal')
    circles(qx[len(qx) // 2:], qy[len(qx) // 2:], 1 / 2, c='coral')
    ax.axis('equal')
    ax.grid()


def display_circle(qx, qy):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # Now, we draw our points with a gradient of colors.
    circles(qx[:len(qx) // 2], qy[:len(qx) // 2], 1 / 2, ec='teal', fc='none')
    circles(qx[len(qx) // 2:], qy[len(qx) // 2:], 1 / 2, ec='coral',fc='none')
    ax.axis('equal')
    ax.grid()


def display_arrow(qx, qy, theta, range='all',step=0.5):
    if range == 'all':
        for i,_ in enumerate(qx):
            plt.arrow(qx[i],qy[i], step*np.cos(theta)[i], step*np.sin(theta)[i], color='grey')
    else:
        for i in range:
            plt.arrow(qx[i],qy[i], step*np.cos(theta)[i], step*np.sin(theta)[i], color='grey')



def display_label(posx, posy):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Now, we draw our points with a gradient of colors.
    ax.scatter(posx[:len(posx) // 2], posy[:len(posx) // 2], linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='teal', cmap=plt.cm.jet,
               label=np.arange(len(posx) // 2))
    ax.scatter(posx[len(posx) // 2:], posy[len(posx) // 2:], linewidths=0.5,
               marker='o', s=5, facecolors='none', edgecolors='coral', cmap=plt.cm.jet,
               label=np.arange(len(posx) // 2, len(posx), 1))

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
    return np.pi * N / (4 * Lx * Ly)


def cal_Pe(sigma, v, D):
    '''
    Calculate the Peclet number
    '''
    t = sigma ** 2 / D
    Pe = t * v / sigma
    return Pe


def draw_gif(data):
    fig, ax = plt.subplots()

    def animate(i):
        fig.clear()
        line = display(data[i][0], data[i][1])
        ax.set_title("iteration " + str(i))
        return line

    ani = FuncAnimation(fig, animate, interval=200, frames=354)

    ani.save("test0.gif", dpi=200, writer=PillowWriter(fps=24))


def generate_gif(folder_name):
    import os
    data = []
    from natsort import humansorted
    for filename in sorted(os.scandir(folder_name), key=lambda s: humansorted(s.path)):
        temp = load(filename.path)
        data.append([temp['qx'], temp['qy']])
    draw_gif(data)
    # os_sorted()


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
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)

    # compute spacing
    init_dist = np.min((x[1] - x[0], y[1] - y[0]))

    # perturb points
    max_movement = (init_dist - min_dist) / 2
    noise = np.random.uniform(low=-max_movement,
                              high=max_movement,
                              size=(len(coords), 2))
    coords += noise

    return coords
