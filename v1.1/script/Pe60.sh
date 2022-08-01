#!/bin/bash
#PBS -N Pe60
#PBS -m a
#PBS -q parallel16
#PBS -k oe
#PBS -t 0-10

cd $HOME/MIPS/v1.1
# number of output data
epoch=500000
# number of iteration in each data
savenum=100
# number of particles
N=10000
# number of grids
M=20
# time step 1e-8
step=500
# simulation size
Llst=(105 110 115 120 125 130 135 140 145 150 155)
L=${Llst[$PBS_ARRAYID]}
# value of Peclet number
Pe=60

python3 main.py $epoch $savenum $N $M $step $L $Pe


# generate figures
folding=$((${N}*31415926/(400000*L*L)))

name=F${folding}P${Pe}T${step}

folderpath=$HOME/MIPS/v1.1/results/${name}
SavePath=$HOME/MIPS/v1.1/fig/${name}/
FinalPath=$HOME/MIPS/v1.1/results/final/${name}.npz
FinalSavePath=$HOME/MIPS/v1.1/fig/final/${name}/

cd $HOME/MIPS/v1.1

python3 gen_fig.py ${FinalPath} ${FinalSavePath}

for file in $folderpath/*; do
  if [[ "${file}" == *0.npz ]]
  then
    python3 gen_fig.py ${file} ${SavePath}
  fi
done

# generate video
picLoadPath=${SavePath}
videoSavePath=$HOME/MIPS/v1.1/video/

python3 gen_gif.py 0 ${epoch} ${name}.avi ${picLoadPath} ${videoSavePath}