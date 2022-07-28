#!/bin/bash
#PBS -N testT
#PBS -m a
#PBS -q parallel8
#PBS -k oe
#PBS -t 0-6

cd $HOME/MIPS/v2.0
# number of output data
# number of iteration in each data
# number of particles
# number of grids
# time step 1e-8
# simulation size
# value of Peclet number
# value of W value 1e-3
tlst=(100 1000 5000 10000 20000 50000 100000)
python3 main.py 5000 500 4900 20 ${tlst[$PBS_ARRAYID]} 80 120 100