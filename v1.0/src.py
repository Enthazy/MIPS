import numpy as np
import matplotlib.pyplot as plt
import math
from time import time
from numba import jit, njit, prange
from numba.typed import List
from matplotlib.animation import FuncAnimation, PillowWriter

from utils import *

NOPYTHON = False
# NOPYTHON = True
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
    :return: the gradient vector dx, dy
    '''
    rx = x/d2   # direction x
    ry = y/d2   # direction y
    k = (1/d2)**3
    s = 24 * (k - 2 * k**2)  # strength
    return s*rx, s*ry


@jit(nopython=NOPYTHON, parallel=True)
def vec_updater(grid, posx, posy, vecx, vecy,
                M, Lx, Ly):
    '''
        Calculate the velocity of particles

    :param grid: list[int], contains the indices of particles in the grids
    :param posx: list[float], x position of particles
    :param posy: list[float], y position of particles
    :param vecx: list[float], x velocity of particles
    :param vecy: list[float], y velocity of particles
    :param M: M**2 is the number of grids
    :param Lx: the size of the box in x
    :param Ly: the size of the box in y
    :return: the velocity vectors
    '''
    #================================================
    # 1. get info for the current grid, find neighbour grid
    #================================================
    for idx_grid in prange(len(grid)):
        # 1.1 get the points in the current grid
        points = grid[idx_grid]

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

                x1 = posx[i]
                x2 = posx[j]
                x_diff = x1 - x2
                if is_bdy: x_diff = min(x_diff, Lx - x1 + x2) # boundary effect
                if  root62n < x_diff < root62: # close enough in x direction

                    y1 = posy[i]
                    y2 = posy[j]
                    y_diff = y1 - y2
                    if is_bdy: y_diff = min(y_diff, Ly - y1 + y2) # boundary effect
                    if root62n < y_diff < root62: # close enough in y direction

                        d2 = x_diff ** 2 + y_diff ** 2
                        if d2 < root32: # distance smaller than 2^1/3
                            #=========================================================
                            # Calculate the interaction
                            #=========================================================
                            vpx, vpy = gradient_reduced_LJPotential(x_diff, y_diff, d2)
                            vecx[i] -= vpx
                            vecy[i] -= vpy
                            vecx[j] += vpx
                            vecy[j] += vpy

            #======================================================
            # 4. Collide Detection in the neighbour grid
            #======================================================
            for idx_j in prange(len(neighbour_points)):
                j = neighbour_points[idx_j] # second particle
                if i == j: continue

                x1 = posx[i]
                x2 = posx[j]
                x_diff = x1 - x2
                if is_bdy: x_diff = min(x_diff, Lx - x1 + x2) # boundary effect
                if  root62n < x_diff < root62: # close enough in x direction

                    y1 = posy[i]
                    y2 = posy[j]
                    y_diff = y1 - y2
                    if is_bdy: y_diff = min(y_diff, Ly - y1 + y2) # boundary effect
                    if root62n < y_diff < root62: # close enough in y direction

                        d2 = x_diff ** 2 + y_diff ** 2
                        if d2 < root32: # distance smaller than 2^1/3
                            #=========================================================
                            # Calculate the interaction
                            #=========================================================
                            vpx, vpy = gradient_reduced_LJPotential(x_diff, y_diff, d2)
                            vecx[i] -= vpx
                            vecy[i] -= vpy
                            vecx[j] += vpx
                            vecy[j] += vpy

def grid_init(M):
    grid = List()
    for i in range(M*M):
        grid.append(List(np.zeros(1).astype(np.int64)))
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
def grid_seperation(grid, posx, posy, M, Lx, Ly):
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
    N = len(posx)
    idx = (posx // (Lx/M)).astype(np.int64)
    idy = (posy // (Ly/M)).astype(np.int64)
    # idx = np.array([M+x if x<0 else x for x in idx])
    # idy = np.array([M+x if x<0 else x for x in idy])
    lst = [[0] for _ in range(M**2)]
    for i in range(N):
        tempy = idy[i]
        tempx = idx[i]
        # there are M+1 edges but only M grids
        if tempy == M: tempy=0
        if tempx == M: tempx=0
        t = np.int64(tempx + tempy*M)
        lst[t].append(np.int64(i))
    for i in range(M**2):
        grid[i] = List(lst[i])
    return grid

@jit(nopython=NOPYTHON)
def dynamics(grid, posx, posy, vecx, vecy, theta, s_x, s_y, s_theta, Pe, M, Lx, Ly):
    vec_updater(grid, posx, posy, vecx, vecy, M, Lx, Ly)
    vecx += Pe * np.cos(theta)
    vecy += Pe * np.sin(theta)
    vecx += sqrt2 * s_x
    vecy += sqrt2 * s_y
    d_theta = sqrt6 * s_theta
    return vecx, vecy, d_theta

@jit(nopython=NOPYTHON)
def updater(step, grid, posx, posy, vecx, vecy, theta, s_x, s_y, s_theta, Pe, M, Lx, Ly):
    dx, dy, d_theta = dynamics(grid, posx, posy, vecx, vecy, theta, s_x, s_y, s_theta, Pe, M, Lx, Ly)
    posx += step * dx
    posy += step * dy
    theta += step * d_theta

@jit(nopython=NOPYTHON)
def run(step, grid, posx, posy, vecx, vecy, theta, Pe, N, M, Lx, Ly):
    s_x = np.random.randn(N)
    s_y = np.random.randn(N)
    s_theta = np.random.randn(N)
    updater(step, grid, posx, posy, vecx, vecy, theta, s_x, s_y, s_theta, Pe, M, Lx, Ly)
    vecx.fill(0)
    vecy.fill(0)
    posx = np.remainder(posx, Lx).astype(np.float32)
    posy = np.remainder(posy, Ly).astype(np.float32)
    theta = np.remainder(theta, 2 * np.pi).astype(np.float32)
    return posx, posy, vecx, vecy, theta