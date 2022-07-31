#!/bin/bash
#PBS -N under_single
#PBS -m a
#PBS -q parallel16
#PBS -k oe

cd $HOME/MIPS/v2.0
# number of output data
# number of iteration in each data
# number of particles
# number of grids
# time step 1e-8
# simulation size
# value of Peclet number
# value of W value 1e-3
python3 main.py 3000 1000 4900 20 100000 80 120 1000