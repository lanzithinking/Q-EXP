#!/bin/bash

# run python script
if [ $# -eq 0 ]; then
	ker_NO=1
	q=1
	whiten=0
	NCG=0
elif [ $# -eq 1 ]; then
	ker_NO="$1"
	q=1
	whiten=0
	NCG=0
elif [ $# -eq 2 ]; then
	ker_NO="$1"
	q="$2"
	whiten=0
	NCG=0
elif [ $# -eq 3 ]; then
	ker_NO="$1"
	q="$2"
	whiten="$3"
	NCG=0
elif [ $# -eq 4 ]; then
	ker_NO="$1"
	q="$2"
	whiten="$3"
	NCG="$4"
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

for mdl_NO in {0..2}
do
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
	sbatch --job-name=${mdl_name}-${ker_name} --output=MAP-${mdl_name}-${ker_name}.log run_MAP.sh ${mdl_NO} ${ker_NO} ${q} ${whiten} ${NCG}
	echo "Job MAP-${mdl_name}-${ker_name} submitted."
done
