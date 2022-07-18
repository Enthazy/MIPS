import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial import voronoi_plot_2d
from test import *


def load(file_name):
    """ Loads the model from numpy file.
    """
    print("Loading from " + file_name)
    return dict(np.load(file_name))


def load_points(savepoint):
    data = load("./results/" + str(savepoint) + ".npz")
    qx = data['qx']
    qy = data['qy']
    # Lx = data['L']
    # Ly = data['L']
    print("=========Load savepoint successfully=========")
    points = np.array([[qx[i], y] for i, y in enumerate(qy)])
    box = (0, 120, 0, 120)
    return points, box


def in_box(towers, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0],
                                         towers[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= towers[:, 1],
                                         towers[:, 1] <= bounding_box[3]))


def voronoi(towers, bounding_box):
    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)
    # Mirror points
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = sp.spatial.Voronoi(points)
    # Filter regions

    vor.filtered_points = points_center
    vor.filtered_regions = [vor.regions[vor.point_region[i]] for i in range(len(points_center))]
    return vor


def centroid_region(vertices):
    # Polygon's signed area
    A = 0
    # Centroid's x
    C_x = 0
    # Centroid's y
    C_y = 0
    for i in range(0, len(vertices) - 1):
        s = (vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1])
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    return np.array([[C_x, C_y]])


def voronoi_plot(vor):
    fig = plt.figure()
    ax = fig.gca()
    # Plot initial points
    ax.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], marker='o', markersize=3, color='coral', ls='')
    # Plot ridges points
    for region in vor.filtered_regions:
        vertices = vor.vertices[region, :]
        ax.plot(vertices[:, 0], vertices[:, 1], marker='o', markersize=1, color='teal', ls='')
    # Plot ridges
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        ax.plot(vertices[:, 0], vertices[:, 1], ls='--', color='k', linewidth='0.5')
    plt.annotate('727', (vor.filtered_points[727, 0], vor.filtered_points[727, 1]))


def voronoi_area(vor):
    area = [ConvexHull(vor.vertices[vor.filtered_regions[i], :]).volume for i, x in enumerate(vor.filtered_regions)]
    return area


def voronoi_histogram(vor):
    fig = plt.figure()
    area = voronoi_area(vor)
    area = np.sort(area)[int(0.0035*len(area)):]
    data = pd.DataFrame({'density': np.divide(1, area)})
    sns.histplot(data, x='density', binwidth=0.002, kde=True)
    plt.grid(alpha=0.5)
    # print(np.argmax(area))


def voronoi_histogram_multi(savepoint, num):
    points1, box = load_points(savepoint)
    area = voronoi_area(voronoi(points1, box))

    for i in range(num-1):
        points2, box = load_points(savepoint - 1*(i+1))
        area2 = voronoi_area(voronoi(points2, box))
        area = np.concatenate((area, area2))

    area = np.sort(area)[int(0.0045*len(area)):]

    fig = plt.figure()
    data = pd.DataFrame({'density': np.divide(1, area)})
    sns.histplot(data, x='density', binwidth=0.002, kde=True, kde_kws={'bw_adjust':1})

    plt.grid(alpha=0.5)


def presenter(savepoint):
    """

    :param savepoint: int
    """
    points, box = load_points(savepoint)
    vor = voronoi(points, box)
    # generate voronoi plot
    # voronoi_plot(vor)

    # generate density histogram
    voronoi_histogram(vor)
    plt.title("Histogram for epoch "+str(savepoint))

    # generate density histogram for multi epoch
    # voronoi_histogram_multi(savepoint, 3)
    # plt.title("Histogram for multi epoch around "+str(savepoint))


def main(filename):
    presenter(int(filename[:-4]))

    import os
    os.makedirs("./fig/", exist_ok=True)
    os.makedirs("./fig/v/", exist_ok=True)

    plt.savefig("./fig/v/" + filename[:-4] + ".png")
    plt.show()
    plt.cla()
    plt.clf()
    plt.close('all')


if __name__ == "__main__":
    main("999.npz")
    # generate_figure(sys.argv[1])
