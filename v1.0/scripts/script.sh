#!/bin/bash
#PBS -N firsttest
#PBS -m abe
#PBS -q standard

cd $HOME/MIPS/scripts

### SAR OTAR Lattice(S/R) rho, trigShape
python3 main_lite.py
