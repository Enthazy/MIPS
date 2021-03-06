import numpy as np
import matplotlib.pyplot as plt
import math
from time import time
from numba import jit, njit, prange
from numba.typed import List


from utils import *

# NOPYTHON = False
NOPYTHON = True
sqrt6 = np.sqrt(6)
sqrt2 = np.sqrt(2)
root32 = 2**(1/3)
root62 = 2**(1/6)
root62n = -1 * 2**(1/6)

@jit(nopython=NOPYTHON)
def gradient_reduced_LJPotential(x, y, d2):
    '''
        calculat the gradient of the FJ potential
        Notice that the Potential is zero for d > 2^1/6 ~ 1.12
        gradient is 24((1/r^6)-2(1/r^12))
    :param x: difference on x between two points
    :param y: difference on y between two points
    :param d2: squared distance d2 = x^2 + y^2
    :return: the gradient ptor dx, dy
    '''
    rx = x/d2   # direction x
    ry = y/d2   # direction y
    k = (1/d2)**3
    s = 24 * (k - 2 * k**2)  # strength
    # assert s < 1e5
    return s*rx, s*ry


@jit(nopython=NOPYTHON, parallel=True)
def p_updater(grid, qx, qy, px, py,
                M, Lx, Ly):
    '''
        Calculate the velocity of particles

    :param grid: list[int], contains the indices of particles in the grids
    :param qx: list[float], x qition of particles
    :param qy: list[float], y qition of particles
    :param px: list[float], x velocity of particles
    :param py: list[float], y velocity of particles
    :param M: M**2 is the number of grids
    :param Lx: the size of the box in x
    :param Ly: the size of the box in y
    :return: the velocity ptors
    '''
    #================================================
    # 1. get info for the current grid, find neighbour grid
    #================================================
    for idx_grid in prange(len(grid)):
        # 1.1 get the points in the current grid
        points = grid[idx_grid]  # list[int] of indices of particles

        # 1.2 the grid location
        indx = idx_grid%M
        indy = idx_grid//M

        # 1.3 Check if the grid is near boundary
        is_bdy = indx==0 or indx==M-1 or indy==0 or indy==M-1

        # 1.4 get the points in the neighbour grid
        neighbour_points = get_cal_range(grid, indx, indy, M)

        #================================================
        # 2. For each point p in the grid, calculate the interaction between
        #================================================
        for idx_i in prange(len(points)):

            i = points[idx_i] # first particle

            #======================================================
            # 3. Collide Detection in the current grid
            #======================================================
            for idx_j in prange(len(points)-idx_i-1):
                j = points[len(points)-idx_j-1] # second particle
                if i == j: continue

                x1 = qx[i]
                x2 = qx[j]

                if is_bdy:  # Boundary effect
                    if x2 > x1 + Lx/2: x2 = x2 - Lx
                    if x1 > x2 + Lx/2: x1 = x1 - Lx
                x_diff = x1 - x2
                
                if  root62n < x_diff < root62: # close enough in x direction

                    y1 = qy[i]
                    y2 = qy[j]
                    if is_bdy:  # Boundary effect
                        if y2 > y1 + Ly/2: y2 = y2 - Ly
                        if y1 > y2 + Ly/2: y1 = y1 - Ly
                    y_diff = y1 - y2
                    if root62n < y_diff < root62: # close enough in y direction

                        d2 = x_diff ** 2 + y_diff ** 2
                        if d2 < root32: # distance smaller than 2^1/3
                            #=========================================================
                            # Calculate the interaction
                            #=========================================================
                            vpx, vpy = gradient_reduced_LJPotential(x_diff, y_diff, d2)
                            px[i] -= vpx
                            py[i] -= vpy
                            px[j] += vpx
                            py[j] += vpy

            #======================================================
            # 4. Collide Detection in the neighbour grid
            #======================================================
            for idx_j in prange(len(neighbour_points)):
                j = neighbour_points[idx_j] # second particle
                if i == j: continue

                x1 = qx[i]
                x2 = qx[j]
                if is_bdy:  # Boundary effect
                    if x2 > x1 + Lx/2: x2 = x2 - Lx
                    if x1 > x2 + Lx/2: x1 = x1 - Lx
                x_diff = x1 - x2
                if  root62n < x_diff < root62: # close enough in x direction

                    y1 = qy[i]
                    y2 = qy[j]
                    if is_bdy:  # Boundary effect
                        if y2 > y1 + Ly/2: y2 = y2 - Ly
                        if y1 > y2 + Ly/2: y1 = y1 - Ly
                    y_diff = y1 - y2
                    if root62n < y_diff < root62: # close enough in y direction

                        d2 = x_diff ** 2 + y_diff ** 2
                        if d2 < root32: # distance smaller than 2^1/3
                            #=========================================================
                            # Calculate the interaction
                            #=========================================================
                            vpx, vpy = gradient_reduced_LJPotential(x_diff, y_diff, d2)
                            # assert vpx < 1e6 and vpy < 1e6
                            px[i] -= vpx
                            py[i] -= vpy
                            px[j] += vpx
                            py[j] += vpy

def grid_init(M):
    grid = List()
    for i in range(M*M):
        grid.append(List(np.zeros(1).astype(np.int32)))
    return grid

@jit(nopython=NOPYTHON)
def get_cal_range(grid, indx, indy, M):
    '''
      For G(ind1, ind2)
      Get the range in the following grid denoted by O. and skip X
              ^
            O O O
      indy  X G O ->
            X X X
            indx
    '''
    # c = list(grid[ind][1:])
    r = list(grid[(indx+1) % M + indy * M][1:])
    tr = list(grid[(indx+1) % M + ((indy+1) % M) * M][1:])
    t = list(grid[indx + ((indy+1) % M) * M][1:])
    tl = list(grid[(indx-1) % M + ((indy+1) % M) * M][1:])
    return r+t+tl+tr

@jit(nopython=NOPYTHON, parallel=True)
def grid_seperation(grid, qx, qy, M, Lx, Ly):
    '''
      For G(indx, indy), the index is $ind = indy * M + indx$
      Get the range in the following grid denoted by O. and skip X

              ^
            X X X
      indy  X X X ->
            X X X
            indx
    '''
    M = int(M)
    N = len(qx)
    idx = (qx // (Lx/M)).astype(np.int32)
    idy = (qy // (Ly/M)).astype(np.int32)
    # idx = np.array([M+x if x<0 else x for x in idx])
    # idy = np.array([M+x if x<0 else x for x in idy])
    lst = [[np.int32(0)] for _ in range(M**2)]
    for i in range(N):
        tempy = idy[i]
        tempx = idx[i]
        # there are M+1 edges but only M grids
        if tempy == M: tempy=0
        if tempx == M: tempx=0
        t = np.int32(tempx + tempy*M)
        lst[t].append(np.int32(i))
    for i in range(M**2):
        grid[i] = List(lst[i])
    return grid

@jit(nopython=NOPYTHON)
def dynamics(grid, qx, qy, px, py, theta, Pe, M, Lx, Ly):
    """
        Calculate the velocity of the particles under
        1.  LJ-Potential
        2.  Self-Proportional force

    :return:
    """
    p_updater(grid, qx, qy, px, py, M, Lx, Ly)
    px += Pe * np.cos(theta)
    py += Pe * np.sin(theta)
    return px, py


@jit(nopython=NOPYTHON)
def updater(step, grid, qx, qy, px, py, theta, s_x, s_y, s_theta, Pe, M, Lx, Ly):
    """
        Update the position and velocity
        Update the Stochastic Brownian motion
    :return:
    """
    sqrt_step = np.sqrt(step)
    # Over-damping
    px.fill(0)
    py.fill(0)

    dx, dy = dynamics(grid, qx, qy, px, py, theta, Pe, M, Lx, Ly)

    qx += step * dx + sqrt2 * sqrt_step * s_x
    qy += step * dy + sqrt2 * sqrt_step * s_y
    theta += sqrt6 * sqrt_step * s_theta


@jit(nopython=NOPYTHON)
def run(step, grid, qx, qy, px, py, theta,
        s_x, s_y, s_theta,
        Pe, N, M, Lx, Ly):
    # generate random variable
    # s_x = np.random.randn(N)
    # s_y = np.random.randn(N)
    # s_theta = np.random.randn(N)

    # update the position q and velocity p
    updater(step, grid, qx, qy, px, py, theta, s_x, s_y, s_theta, Pe, M, Lx, Ly)

    # fold back the periodic points
    qx = np.remainder(qx, Lx).astype(np.float32)
    qy = np.remainder(qy, Ly).astype(np.float32)
    theta = np.remainder(theta, 2 * np.float32(np.pi)).astype(np.float32)
    return qx, qy, px, py, theta