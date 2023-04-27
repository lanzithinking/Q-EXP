#!/bin/bash

# run python script
if [ $# -eq 0 ]; then
	alg_NO=0
	mdl_NO=1
	ker_NO=1
	q=1
elif [ $# -eq 1 ]; then
	alg_NO="$1"
	mdl_NO=1
	ker_NO=1
	q=1
elif [ $# -eq 2 ]; then
	alg_NO="$1"
	mdl_NO="$2"
	ker_NO=1
	q=1
elif [ $# -eq 3 ]; then
	alg_NO="$1"
	mdl_NO="$2"
	ker_NO="$3"
	q=1
elif [ $# -eq 4 ]; then
	alg_NO="$1"
	mdl_NO="$2"
	ker_NO="$3"
	q="$4"
fi

if [ ${alg_NO} -eq 0 ]; then
	alg_name='wpCN'
elif [ ${alg_NO} -eq 1 ]; then
	alg_name='winfMALA'
elif [ ${alg_NO} -eq 2 ]; then                                                  
    alg_name='winfHMC'
elif [ ${alg_NO} -eq 3 ]; then
	alg_name='winfmMALA'
elif [ ${alg_NO} -eq 4 ]; then                                                  
    alg_name='winfmHMC'
else
	echo "Wrong args!"
	exit 0
fi

if [ ${mdl_NO} -eq 0 ]; then
	mdl_name='gp'
elif [ ${mdl_NO} -eq 1 ]; then
	mdl_name='bsv'
elif [ ${mdl_NO} -eq 2 ]; then                                                  
    mdl_name='qep'
else
	echo "Wrong args!"
	exit 0
fi

if [ ${ker_NO} -eq 0 ]; then
	ker_name='covf'
elif [ ${ker_NO} -eq 1 ]; then
	ker_name='serexp'
elif [ ${ker_NO} -eq 2 ]; then                                                  
    ker_name='graphL'
else
	echo "Wrong args!"
	exit 0
fi

for seed_NO in {2022..2112..10}
do
	sbatch --job-name=${alg_NO}-${mdl_name}-${seed_NO} --output=${alg_NO}-${mdl_name}-${ker_name}-${seed_NO}.log run_wgeoinfMC.sh ${alg_NO} ${seed_NO} ${mdl_NO} ${ker_NO} ${q}
	echo "Job ${alg_NO}-${mdl_name}-${ker_name}-${seed_NO} submitted."
done
