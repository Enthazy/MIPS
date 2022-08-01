import numpy as np
from numba.typed import List
from numba import jit, prange
from utils import *


# @jit(nopython=True)
def grid_init(M):
    grid = List()
    for i in range(M * M):
        grid.append(List(np.zeros(1).astype(np.int64)))
    return grid


# @jit(nopython=True)
def get_cal_range(grid, indx, indy, M):
    """
      For G(ind1, ind2)
      Get the range in the following grid denoted by O. and skip X
              ^
            O O O
      indy  X G O ->
            X X X
            indx
    """
    # c = list(grid[ind][1:])
    r = list(grid[(indx + 1) % M + indy * M][1:])
    tr = list(grid[(indx + 1) % M + ((indy + 1) % M) * M][1:])
    t = list(grid[indx + ((indy + 1) % M) * M][1:])
    tl = list(grid[(indx - 1) % M + ((indy + 1) % M) * M][1:])
    return r + t + tl + tr


# @jit(nopython=True, parallel=True)
def grid_seperation(grid, qx, qy, M, Lx, Ly):
    """
      For G(indx, indy), the index is $ind = indy * M + indx$
      Get the range in the following grid denoted by O. and skip X

              ^
            X X X
      indy  X X X ->
            X X X
            indx
    """
    M = int(M)
    N = len(qx)
    idx = (qx // (Lx / M)).astype(np.int64)
    idy = (qy // (Ly / M)).astype(np.int64)
    # idx = np.array([M+x if x<0 else x for x in idx])
    # idy = np.array([M+x if x<0 else x for x in idy])
    lst = [[0] for _ in range(M ** 2)]
    for i in range(N):
        tempy = idy[i]
        tempx = idx[i]
        # there are M+1 edges but only M grids
        if tempy == M: tempy = 0
        if tempx == M: tempx = 0
        t = np.int64(tempx + tempy * M)
        lst[t].append(np.int64(i))
    for i in range(M ** 2):
        grid[i] = List(lst[i])
    return grid


# @jit(nopython=True)
def gradient_reduced_LJPotential(x, y, d2):
    """
        calculat the gradient of the FJ potential
        Notice that the Potential is zero for d > 2^1/6 ~ 1.12
        gradient is 24((1/r^6)-2(1/r^12))
    :param x: difference on x between two points
    :param y: difference on y between two points
    :param d2: squared distance d2 = x^2 + y^2
    :return: the gradient ptor dx, dy
    """
    rx = x / d2  # direction x
    ry = y / d2  # direction y
    k = (1 / d2) ** 3
    s = 24 * (k - 2 * k ** 2)  # strength
    return s * rx, s * ry


# @jit(nopython=True, parallel=True)
def interaction_calculator(dpx, dpy,
                           grid, qx, qy,
                           M, Lx, Ly):
    """
        Calculate the interactions under LJ-potential
        output the velocity gradients dpx, dpy

    :param grid: list[int], contains the indices of particles in the grids
    :param qx: list[float], x qition of particles
    :param qy: list[float], y qition of particles
    :param dpx: list[float], x velocity of particles
    :param dpy: list[float], y velocity of particles
    :param M: M**2 is the number of grids
    :param Lx: the size of the box in x
    :param Ly: the size of the box in y
    :return: the velocity ptors
    """
    # ================================================
    # 1. get info for the current grid, find neighbour grid
    # ================================================
    root32 = np.float32(np.power(2,1/3))
    root62 = np.float32(np.power(2,1/6))
    root62n = np.float32(-1*root62)
    for idx_grid in prange(len(grid)):
        # 1.1 get the points in the current grid
        points = grid[idx_grid]  # list[int] of indices of particles

        # 1.2 the grid location
        indx = idx_grid % M
        indy = idx_grid // M

        # 1.3 Check if the grid is near boundary
        is_bdy = indx == 0 or indx == M - 1 or indy == 0 or indy == M - 1

        # 1.4 get the points in the neighbour grid
        neighbour_points = get_cal_range(grid, indx, indy, M)

        # ================================================
        # 2. For each point p in the grid, calculate the interaction between
        # ================================================
        for idx_i in prange(len(points)):

            i = points[idx_i]  # first particle

            # ======================================================
            # 3. Collide Detection in the current grid
            # ======================================================
            for idx_j in prange(len(points) - idx_i - 1):
                j = points[len(points) - idx_j - 1]  # second particle
                if i == j: continue

                x1 = qx[i]
                x2 = qx[j]

                if is_bdy:  # Boundary effect
                    if x2 > x1 + Lx / 2: x2 = x2 - Lx
                    if x1 > x2 + Lx / 2: x1 = x1 - Lx
                x_diff = x1 - x2

                if root62n < x_diff < root62:  # close enough in x direction

                    y1 = qy[i]
                    y2 = qy[j]
                    if is_bdy:  # Boundary effect
                        if y2 > y1 + Ly / 2: y2 = y2 - Ly
                        if y1 > y2 + Ly / 2: y1 = y1 - Ly
                    y_diff = y1 - y2
                    if root62n < y_diff < root62:  # close enough in y direction

                        d2 = x_diff ** 2 + y_diff ** 2
                        if d2 < root32:  # distance smaller than 2^1/3
                            # =========================================================
                            # Calculate the interaction
                            # =========================================================
                            vpx, vpy = gradient_reduced_LJPotential(x_diff, y_diff, d2)
                            dpx[i] -= vpx
                            dpy[i] -= vpy
                            dpx[j] += vpx
                            dpy[j] += vpy

            # ======================================================
            # 4. Collide Detection in the neighbour grid
            # ======================================================
            for idx_j in prange(len(neighbour_points)):
                j = neighbour_points[idx_j]  # second particle
                if i == j: continue

                x1 = qx[i]
                x2 = qx[j]
                if is_bdy:  # Boundary effect
                    if x2 > x1 + Lx / 2: x2 = x2 - Lx
                    if x1 > x2 + Lx / 2: x1 = x1 - Lx
                x_diff = x1 - x2
                if root62n < x_diff < root62:  # close enough in x direction

                    y1 = qy[i]
                    y2 = qy[j]
                    if is_bdy:  # Boundary effect
                        if y2 > y1 + Ly / 2: y2 = y2 - Ly
                        if y1 > y2 + Ly / 2: y1 = y1 - Ly
                    y_diff = y1 - y2
                    if root62n < y_diff < root62:  # close enough in y direction

                        d2 = x_diff ** 2 + y_diff ** 2
                        if d2 < root32:  # distance smaller than 2^1/3
                            # =========================================================
                            # Calculate the interaction
                            # =========================================================
                            vpx, vpy = gradient_reduced_LJPotential(x_diff, y_diff, d2)
                            dpx[i] -= vpx
                            dpy[i] -= vpy
                            dpx[j] += vpx
                            dpy[j] += vpy


# @jit(nopython=True)
def p_gradient_calculator(grid, qx, qy, qtheta,
                          px, py,
                          Pe, M, Lx, Ly):
    """
        Calculate the velocity of the particles under
        1.  LJ-Potential
        2.  Self-Proportional force

    :return:
    """

    # initial a gradient vector
    dpx = np.zeros_like(px)
    dpy = np.zeros_like(py)

    interaction_calculator(dpx, dpy, grid, qx, qy, M, Lx, Ly)

    dpx += Pe * np.cos(qtheta) - px
    dpy += Pe * np.sin(qtheta) - py

    return dpx, dpy


# @jit(nopython=True)
def p_updater(dpx, dpy,
              px, py, ptheta,
              s_x, s_y, s_theta,
              step, W):
    """
        Calculate the velocity of the particles under
        1.  LJ-Potential
        2.  Self-Proportional force

    :return:
    """
    sqrt2 = np.float32(np.sqrt(2))
    sqrt6 = np.float32(np.sqrt(6))
    sqrt_step = np.sqrt(step)

    px += (step * dpx + sqrt2 * sqrt_step * s_x) / W
    py += (step * dpy + sqrt2 * sqrt_step * s_y) / W
    ptheta += sqrt6 * sqrt_step * s_theta / (3*W/8)
    return px, py, ptheta


# @jit(nopython=True)
def q_gradient_calculator(px, py, ptheta):
    """
        Update the position and velocity
        Update the Stochastic Brownian motion

    :return:
    """
    dqx = px
    dqy = py
    dqtheta = ptheta
    return dqx, dqy, dqtheta


# @jit(nopython=True)
def q_updater(dqx, dqy, dqtheta,
              qx, qy, qtheta,
              step):
    """
        Update the position and velocity
        Update the Stochastic Brownian motion
    :return:
    """

    qx += step * dqx
    qy += step * dqy
    qtheta += step * dqtheta


# @jit(nopython=True)
def euler_updater(qx, qy, qtheta, px, py, ptheta,
                  ax, ay,
                  s_x, s_y, s_theta,
                  grid, step, Pe, W, M, Lx, Ly):

    # Calculate dq, dp
    dpx, dpy = p_gradient_calculator(grid, qx, qy, qtheta,
                                              px, py,
                                              Pe, M, Lx, Ly)

    dqx, dqy, dqtheta = q_gradient_calculator(px, py, ptheta)

    # Update q, p
    p_updater(dpx, dpy,
              px, py, ptheta,
              s_x, s_y, s_theta,
              step, W)

    q_updater(dqx, dqy, dqtheta,
              ax, ay, qtheta,
              step)


# @jit(nopython=True)
def run(qx, qy, qtheta, px, py, ptheta,
        ax, ay,
        s_x, s_y, s_theta,
        grid, step, Pe, W, M, Lx, Ly):

    euler_updater(qx, qy, qtheta, px, py, ptheta,
                  ax, ay,
                  s_x, s_y, s_theta,
                  grid, step, Pe, W, M, Lx, Ly)

    # fold back the periodic points
    qx = np.remainder(ax, Lx)
    qy = np.remainder(ay, Ly)
    # qtheta = np.remainder(qtheta, 2 * np.float32(np.pi)).astype(np.float32)
    return qx, qy, qtheta, px, py, ptheta, ax, ay