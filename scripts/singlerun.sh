#!/bin/bash
#PBS -N over_damped_single
#PBS -m a
#PBS -q standard
#PBS -k oe

cd $HOME/MIPS/v1.0
# number of output data
# number of iteration in each data
# number of particles
# number of grids
# simulation size
# value of Peclet number
python3 main.py 1000 1000 4900 20 80 120
