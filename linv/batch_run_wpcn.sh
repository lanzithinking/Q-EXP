#!/bin/bash

# run python script
if [ $# -eq 0 ]; then
	mdl_NO=2
	q=1
elif [ $# -eq 1 ]; then
	mdl_NO="$1"
	q=1
elif [ $# -eq 2 ]; then
	mdl_NO="$1"
	q="$2"
fi

if [ ${mdl_NO} -eq 0 ]; then
	mdl_name="gp"
elif [ ${mdl_NO} -eq 1 ]; then
	mdl_name="qep"
elif [ ${mdl_NO} -eq 2 ]; then                                                  
    mdl_name="bsv"
else
	echo "Wrong args!"
	exit 0
fi

for seed_NO in {2022..2112..10}
do
	sbatch --job-name=${mdl_name}-${seed_NO} --output=wpcn-${mdl_name}-${seed_NO}.log run_wpcn.sh ${seed_NO} ${mdl_NO} ${q}
	echo "Job ESS-${mdl_name}-${seed_NO} submitted."
done
