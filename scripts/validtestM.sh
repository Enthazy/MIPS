#!/bin/bash
#PBS -N od_multi
#PBS -m a
#PBS -q parallel16
#PBS -k o
#PBS -t 0-5

cd $HOME/MIPS/v1.0
# number of output data
# number of iteration in each data
# number of particles
# number of grids
# simulation size
# value of Peclet number
echo $PBS_ARRYID
python3 main.py 10 1000 4900 $((10 + (5 * $PBS_ARRAYID))) 80 120
