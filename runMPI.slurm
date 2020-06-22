#!/bin/bash

#SBATCH --job-name MB_MPI
#SBATCH --error errors_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user barbier4@dei.unipd.it
#SBATCH --mail-type END,FAIL
#SBATCH --partition allgroups
#SBATCH --ntasks 32
#SBATCH --cpus-per-task 1
#SBATCH --mem 1000M 
#SBATCH --time 00:15:00

cd $SLURM_SUBMIT_DIR

srun singularity exec --bind miaCartella:/mnt/ ATSPP/container.sif mpirun -np 32 python3 /mnt/examMTMPI.py esc128 /mnt
