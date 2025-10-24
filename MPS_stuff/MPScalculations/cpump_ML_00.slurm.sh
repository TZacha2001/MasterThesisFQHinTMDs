#!/bin/bash
#SBATCH --job-name=cpump_ML_00
#SBATCH --chdir=./
#SBATCH --output=./logfiles/cpump_ML_00.%A_%3a.out  # %J=jobid.step, %N=node.
#
# To support getting emails, adjust the following two lines and remove the `# `,i.e. make them start with `#SBATCH `
#SBATCH --mail-type ALL  # or `fail,end`, but it's not recommended
#SBATCH --mail-user fabian.pichler@tum.de  # adjust...
# NOTE: use ONLY YOUR UNIVERSITY EMAIL, DON'T USE/FORWARD EMAIL to other email providers like gmail.com!
# You can get a lot of emails from the cluster, and other email providers then sometimes mark the whole university as sending spam.
# This might results in your professor not being able to write emails to his friends anymore...
#SBATCH --time=0-38:02:00
#SBATCH --mem=2G
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --array=1-6

set -e  # abort whole script if any command fails

# === prepare the environement as necessary ===
# module load python/3.7
# conda activate tenpy


# use SLURM_CPUS_PER_TASK, if not set default to SLURM_CPUS_ON_NODE
USE_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE}}
if [ -z "$USE_NUM_THREADS" ]
then
	USE_NUM_THREADS="$(nproc --all)"
	echo "WARNING: SLURM_CPUS_ON_NODE not set! Using all cores on machine, NTHREADS=$USE_NUM_THREADS"
fi
# When requesting --cpus-per-task 32 on nodes with CPU hyperthreading,
# slurm will allocate the job 32 threads = 16 physical cores x 2 (hyper)threads per core.
# For many numerical applications, e.g BLAS/LAPACK functions like matrix diagonalization, 
# it is better to ignore hyperthreading and rather set NUM_THREADS to the number of physical cores.
# Hence we divide by 2 here:
USE_NUM_THREADS=$(($USE_NUM_THREADS / 2 ))
export OMP_NUM_THREADS=$USE_NUM_THREADS  # number of CPUs per node, total for all the tasks below.
export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS=$USE_NUM_THREADS  # number of CPUs per node, total for all the tasks below.
export NUMBA_NUM_THREADS=$USE_NUM_THREADS

echo "Running task $SLURM_ARRAY_TASK_ID specified in logfiles/cpump_ML_00.config.pkl on $HOSTNAME at $(date) with $USE_NUM_THREADS threads"
python /home/t30/kna/ge54yin/Documents/MasterThesisFQHinTMDs/MPS_stuff/MPScalculations/cluster_jobs.py run logfiles/cpump_ML_00.config.pkl $SLURM_ARRAY_TASK_ID
# if you want to redirect output to file, you can append the following to the line above:
#     &> "cpump_ML_00.task_$SLURM_ARRAY_TASK_ID.out"
echo "finished at $(date)"
