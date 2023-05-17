#!/bin/bash
 
#SBATCH -N 1                        # number of compute nodes
#SBATCH -c 5                        # number of "tasks" (cores)
#SBATCH --mem=64G                   # GigaBytes of memory required (per node)

#SBATCH -p parallel                 # partition 
#SBATCH -q normal                   # QOS

#SBATCH -t 1-12:00                  # wall time (D-HH:MM)
##SBATCH -A slan7                   # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o %x.log                   # STDOUT (%j = JobId)
#SBATCH -e %x.err                   # STDERR (%j = JobId)
#SBATCH --mail-type=END,FAIL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=slan@asu.edu    # send-to address

# load environment
# module load python/3.7.1
module load anaconda3/2020.2

# go to working directory
cd ~/Projects/qEP/code/CT_torso

# run python script
if [ $# -eq 0 ]; then
	alg_NO=0
	seed_NO=2022
	mdl_NO=0
	ker_NO=1
	q=1
elif [ $# -eq 1 ]; then
	alg_NO="$1"
	seed_NO=2022
	mdl_NO=0
	ker_NO=1
	q=1
elif [ $# -eq 2 ]; then
	alg_NO="$1"
	seed_NO="$2"
	mdl_NO=0
	ker_NO=1
	q=1
elif [ $# -eq 3 ]; then
	alg_NO="$1"
	seed_NO="$2"
	mdl_NO="$3"
	ker_NO=1
	q=1
elif [ $# -eq 4 ]; then
	alg_NO="$1"
	seed_NO="$2"
	mdl_NO="$3"
	ker_NO="$4"
	q=1
elif [ $# -eq 5 ]; then
	alg_NO="$1"
	seed_NO="$2"
	mdl_NO="$3"
	ker_NO="$4"
	q="$5"
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

python -u run_ct_wgeoinfMC.py ${alg_NO} ${seed_NO} ${mdl_NO} ${ker_NO} ${q} #> ${alg_name}_J${q}.log
# sbatch --job-name=${alg_NO}-${mdl_name}-${seed_NO} --output=${alg_NO}-${mdl_name}-${seed_NO}.log run_wgeoinfMC.sh