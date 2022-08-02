#!/bin/bash
#PBS -N smallW
#PBS -q parallel16
#PBS -k oe
#PBS -t 0-7

cd $HOME/MIPS/v3
# number of output data
epoch=30000
# number of iteration in each data
savenum=100
# number of particles
N=4900
# number of grids
M=20
# time step 1e-8
step=100
# simulation size
L=90
# value of Peclet number
Pe=125
# value of W value 1e-5
tlst=(1 10 100 200 500 1000 2000 5000)
W=${tlst[$PBS_ARRAYID]}

python3 main.py $epoch $savenum $N $M $step $L $Pe $W


# generate figures
folding=$((${N}*31415926/(400000*L*L)))

name=F${folding}P${Pe}W${W}T${step}

folderpath=$HOME/MIPS/v3/results/${name}
SavePath=$HOME/MIPS/v3/fig/${name}/
FinalPath=$HOME/MIPS/v3/results/final/${name}.npz
FinalSavePath=$HOME/MIPS/v3/fig/final/${name}/

cd $HOME/MIPS/v3

python3 gen_fig.py ${FinalPath} ${FinalSavePath}

for file in $folderpath/*; do
  python3 gen_fig.py ${file} ${SavePath}
done

# generate video
picLoadPath=${SavePath}
videoSavePath=$HOME/MIPS/v3/video/

python3 gen_gif.py 0 ${epoch} ${name}.avi ${picLoadPath} ${videoSavePath}