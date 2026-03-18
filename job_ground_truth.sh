#!/bin/bash
#SBATCH --job-name=ground_truth
#SBATCH --output=ground_truth_%j.log
#SBATCH --error=ground_truth_%j.err
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:30:00

source /etc/profile
module load cmake/3.25.1 mpi/openmpi-4.1.5 gcc/13-2-0

cd "${SLURM_SUBMIT_DIR}"

EXE="${SLURM_SUBMIT_DIR}/build/release/darcy_forward"
INPUT="${SLURM_SUBMIT_DIR}/parameters_ground_truth.json"

# Ensure output directory exists
mkdir -p "${SLURM_SUBMIT_DIR}/output/ground_truth"

echo "Running ground truth forward solve with ${SLURM_NTASKS} MPI ranks"
echo "Executable: ${EXE}"
echo "Parameters: ${INPUT}"

mpirun --bind-to core --map-by core -np ${SLURM_NTASKS} ${EXE} ${INPUT}

RC=$?
echo
echo "Ground truth forward solve finished with exit code ${RC} at: $(date)"
ls -lh output/ground_truth/*.npy 2>/dev/null
