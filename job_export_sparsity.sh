#!/bin/bash
#SBATCH --job-name=export_sparsity
#SBATCH --output=export_sparsity_%j.log
#SBATCH --error=export_sparsity_%j.err
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:10:00

source /etc/profile
module load cmake/3.25.1 mpi/openmpi-4.1.5 gcc/13-2-0

# SLURM_SUBMIT_DIR is the directory from which sbatch was called
cd "${SLURM_SUBMIT_DIR}"

EXE="${SLURM_SUBMIT_DIR}/build/release/export_sparsity"
INPUT="${SLURM_SUBMIT_DIR}/parameters_export_sparsity.json"

echo "Running export_sparsity with ${SLURM_NTASKS} MPI ranks"
echo "Executable: ${EXE}"
echo "Parameters: ${INPUT}"
echo "Output dir: $(pwd)"

mpirun --bind-to core --map-by core -np ${SLURM_NTASKS} ${EXE} ${INPUT}

RC=$?
echo
echo "export_sparsity finished with exit code ${RC} at: $(date)"
ls -lh rf_sparsity_*.npy rf_A_kappa_values.npy rf_mass_lumped_diag.npy 2>/dev/null

if [ ${RC} -eq 0 ]; then
  echo
  echo "Computing initial variational parameters..."
  python "${SLURM_SUBMIT_DIR}/compute_prior_init.py" --target-variance 0.1 --mean 0.1 --output initial_variational_params_inverse.npy
  echo "Done. Output:"
  ls -lh initial_variational_params_inverse.npy 2>/dev/null
fi
