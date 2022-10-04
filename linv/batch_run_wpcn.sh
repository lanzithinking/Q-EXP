#!/bin/bash

# run python script
if [ $# -eq 0 ]; then
	mdl_NO=1
	ker_NO=1
	q=1
elif [ $# -eq 1 ]; then
	mdl_NO="$1"
	ker_NO=1
	q=1
elif [ $# -eq 2 ]; then
	mdl_NO="$1"
	ker_NO="$2"
	q=1
elif [ $# -eq 3 ]; then
	mdl_NO="$1"
	ker_NO="$2"
	q="$3"
fi
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
	sbatch --job-name=${mdl_name}-${seed_NO} --output=wpCN-${mdl_name}-${ker_name}-${seed_NO}.log run_wpCN.sh ${seed_NO} ${mdl_NO} ${ker_NO} ${q}
	echo "Job ESS-${mdl_name}-${ker_name}-${seed_NO} submitted."
done
