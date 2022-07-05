#!/bin/bash
#PBS -N od_multi
#PBS -m a
#PBS -q parallel16
#PBS -k o
#PBS -t 0-9

cd $HOME/MIPS/v1.0
# number of output data
# number of iteration in each data
# number of particles
# number of grids
# simulation size
# value of Peclet number
echo $PBS_ARRYID
echo 120
python3 main.py 800 1000 4900 20 $((75 + (5 * $PBS_ARRAYID))) 120