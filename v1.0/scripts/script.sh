#!/bin/bash
#PBS -N firsttest
#PBS -m abe
#PBS -q standard

cd $HOME/MIPS/v1.0

python3 main_lite.py
