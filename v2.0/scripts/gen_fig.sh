#!/bin/bash
#PBS -N gen_fig
#PBS -m a
#PBS -q standard
#PBS -k oe

folderpath=$HOME/MIPS/v2.0/results/F60P120W1000T100000
savepath=$HOME/MIPS/v2.0/fig/F60P120W1000T100000/

cd $HOME/MIPS/v2.0

for file in $folderpath/*; do
  python3 gen_fig.py $file $savepath
done