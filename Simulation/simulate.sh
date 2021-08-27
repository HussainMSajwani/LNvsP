d=$1
h2s=$2
i=$3
dir=Simulation/data/d_$d/h2s_$h2s/sim_$i
mkdir -p $dir

make sim chr=22 d=$d n=600 h2s=$h2s dc=10 dir=$dir > $dir/log.txt