import matplotlib.pyplot as plt
import numpy as np
import sys

def load(file_name):
    """ Loads the model from numpy file.
    """
    print("Loading from " + file_name)
    return dict(np.load(file_name))

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


def display_circle(qx, qy):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # Now, we draw our points with a gradient of colors.
    circles(qx[:len(qx) // 2], qy[:len(qx) // 2], 1 / 2, ec='teal', fc='none')
    circles(qx[len(qx) // 2:], qy[len(qx) // 2:], 1 / 2, ec='coral',fc='none')
    ax.axis('equal')
    ax.grid()

def display_ball(qx, qy):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # Now, we draw our points with a gradient of colors.
    circles(qx[:len(qx) // 2], qy[:len(qx) // 2], 1 / 2, c='teal')
    circles(qx[len(qx) // 2:], qy[len(qx) // 2:], 1 / 2, c='coral')
    ax.axis('equal')
    ax.grid()


def display_arrow(qx, qy, theta, range='all',step=0.5):
    if range == 'all':
        for i,_ in enumerate(qx):
            plt.arrow(qx[i],qy[i], step*np.cos(theta)[i], step*np.sin(theta)[i], color='grey')
    else:
        for i in range:
            plt.arrow(qx[i],qy[i], step*np.cos(theta)[i], step*np.sin(theta)[i], color='grey')

if __name__ == "__main__":
    filepath = sys.argv[1]
    savepath = sys.argv[2]
    # filepath = './results/F40P120T400/999.npz'
    # savepath = './fig/F40P120T400/'
    data = load(filepath)
    qx = data['qx']
    qy = data['qy']
    epoch = data['epoch']
    qtheta = data['qtheta']

    display_circle(qx, qy)
    display_arrow(qx, qy, qtheta)

    import os
    os.makedirs(savepath, exist_ok=True)

    plt.xlim(0, data['L'])
    plt.ylim(0, data['L'])
    plt.savefig(savepath + str(epoch) + ".png")