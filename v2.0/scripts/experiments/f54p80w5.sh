#!/bin/bash
#PBS -N f54p80w5
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
python3 main.py 1000 1000 10000 20 100 120 80 5