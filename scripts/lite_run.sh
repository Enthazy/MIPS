#!/bin/bash
#PBS -N over_damped_single
#PBS -m a
#PBS -q standard
#PBS -k oe

cd $HOME/MIPS/v1.0
python3 main_lite.py