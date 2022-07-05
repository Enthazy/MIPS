#!/bin/bash
#PBS -N moduletest
#PBS -m a
#PBS -q standard
#PBS -k oe

cd $HOME/MIPS/tests
python3 moduletest.py