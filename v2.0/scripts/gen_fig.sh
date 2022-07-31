#!/bin/bash
#PBS -N gen_fig
#PBS -m a
#PBS -q standard
#PBS -k oe

folder=F60P120W1000T100000
folderpath=$HOME/MIPS/v2.0/results/$folder
savepath=$HOME/MIPS/v2.0/fig/$folder/

cd $HOME/MIPS/v2.0

for file in $folderpath/*; do
  python3 gen_fig.py $file $savepath
done