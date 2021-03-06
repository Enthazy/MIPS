#!/bin/bash
#PBS -N lowWtest
#PBS -m a
#PBS -q parallel16
#PBS -k oe
#PBS -t 0-7

cd $HOME/MIPS/v2.0
# number of output data
epoch=50000
# number of iteration in each data
savenum=10
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
# value of W value 1e-5
W=1

python3 main.py $epoch $savenum $N $M $step $L $Pe $W

folding=$((${N}*31415926/(400000*L*L)))

name=F${folding}P${Pe}W${W}T${step}

folderpath=$HOME/MIPS/v2.0/results/${name}
savepath=$HOME/MIPS/v2.0/fig/${name}/
finalpath=$HOME/MIPS/v2.0/results/final/${name}.npz
finalsavepath=$HOME/MIPS/v2.0/fig/final/${name}/

cd $HOME/MIPS/v2.0

python3 gen_fig.py ${finalpath} ${finalsavepath}

for file in $folderpath/*; do
  python3 gen_fig.py ${file} ${savepath}
done