#!/bin/bash
### Specify job parameters
#SBATCH -J trining_lrp # name of the job
#SBATCH -o training_lrp.o%j
#SBATCH -t 5:00:00 # time requested
#SBATCH -N 1 -n 2 # total number of nodes and processes

### Tell SLURM which account to charge this job to
#SBATCH -A vilalta  #Allocation_AWARD_ID
#SBATCH --mem=32GB #Memory size from nodes
python training_test.py
