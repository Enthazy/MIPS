#!/bin/bash
#PBS -N lite_single
#PBS -m a
#PBS -q parallel4
#PBS -k o

cd $HOME/MIPS/v1.0
python3 main_lite.py