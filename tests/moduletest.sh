#!/bin/bash
#PBS -N over_damped_single
#PBS -m a
#PBS -q standard
#PBS -k oe

cd $HOME/MIPS/tests
python3 moduletest.py