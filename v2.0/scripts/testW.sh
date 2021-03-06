#!/bin/bash
#PBS -N testW
#PBS -m a
#PBS -q parallel16
#PBS -k oe
#PBS -t 0-9

cd $HOME/MIPS/v2.0
# number of output data
# number of iteration in each data
# number of particles
# number of grids
# time step 1e-8
# simulation size
# value of Peclet number
# value of W value 1e-3
wlst=(1 2 5 10 20 50 100 200 500 1000)
python3 main.py 100000 500 4900 20 100 80 120 ${wlst[$PBS_ARRAYID]}