#!/bin/bash
#PBS -N template
#PBS -m a
#PBS -q parallel16
#PBS -k oe
#PBS -t 0

cd $HOME/MIPS/v1.1
# number of output data
epoch=3
# number of iteration in each data
savenum=3
# number of particles
N=10000
# number of grids
M=20
# time step 1e-8
tlst=(1 10 100 500 1000 5000 10000 50000)
step=${tlst[$PBS_ARRAYID]}
# simulation size
L=120
# value of Peclet number
Pe=120

python3 main.py $epoch $savenum $N $M $step $L $Pe


# generate figures
folding=$((${N}*31415926/(400000*L*L)))

name=F${folding}P${Pe}T${step}

folderpath=$HOME/MIPS/v3/results/${name}
SavePath=$HOME/MIPS/v3/fig/${name}/
FinalPath=$HOME/MIPS/v3/results/final/${name}.npz
FinalSavePath=$HOME/MIPS/v3/fig/final/${name}/

cd $HOME/MIPS/v1.1

python3 gen_fig.py ${FinalPath} ${FinalSavePath}

for file in $folderpath/*; do
  python3 gen_fig.py ${file} ${SavePath}
done

# generate video
picLoadPath=${SavePath}
videoSavePath=$HOME/MIPS/v3/video/

python3 gen_gif.py 0 ${epoch} ${name}.avi ${picLoadPath} ${videoSavePath}