#!/bin/bash
#PBS -N gen_fig
#PBS -m a
#PBS -q standard
#PBS -k oe

def folderpath=$HOME/MIPS/v2.0/results/final_states

for file in folderpath/*; do
  echo $file
done