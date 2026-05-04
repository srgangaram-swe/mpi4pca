#!/bin/bash
#
# run_pca_easley.sbatch — SLURM batch for Easley to run MPI PCA and plotting
#
# Submit with: sbatch run_pca_easley.sbatch
#
#SBATCH -J pca-mpi
#SBATCH -o pca-mpi-%j.out
#SBATCH -e pca-mpi-%j.err
#SBATCH -t 00:45:00
#SBATCH -p h100           # change if needed (e.g., standard, debug)
##SBATCH -A mpi4pca      # uncomment/set if Easley requires accounting
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive

# --- Modules (customize for Easley) ---
if command -v module >/dev/null 2>&1; then
  module purge
  module load python/3.10 2>/dev/null || true
  module load openmpi 2>/dev/null || module load mpi 2>/dev/null || true
fi

# --- Parameters ---
export NP_LIST="${NP_LIST:-1 2 4 8 16 32}"   # scaling sweep
export ROWS="${ROWS:-5000000}"
export COLS="${COLS:-256}"
export K="${K:-16}"
export OUTBASE="${OUTBASE:-pca_out}"
export DTYPE="${DTYPE:-float64}"
export STANDARDIZE="${STANDARDIZE:-0}"
export WHITEN="${WHITEN:-0}"
export DIMS="${DIMS:-2}"
export RANK_PLOT="${RANK_PLOT:-0}"

# --- Sanity ---
if [[ ! -f "mpi_pca.py" || ! -f "data_vis.py" ]]; then
  echo "ERROR: need mpi_pca.py and data_vis.py in $(pwd)"; exit 1; fi
if [[ ! -f "run_pca_and_vis.sh" ]]; then
  echo "ERROR: need run_pca_and_vis.sh in $(pwd)"; exit 1; fi
chmod +x run_pca_and_vis.sh

echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname)"
echo "NP_LIST=${NP_LIST}"

# --- Shared env for perf ---
export VENVDIR="${VENVDIR:-.venv}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

TIMINGS_CSV="timings_${SLURM_JOB_ID}.csv"
echo "np,load,center,cov_allreduce,eig,project,total" > "${TIMINGS_CSV}"

for NP in ${NP_LIST}; do
  OUTDIR="${OUTBASE}_np${NP}"
  echo "=== NP=${NP} OUTDIR=${OUTDIR} ==="
  ./run_pca_and_vis.sh -n "${NP}" --rows "${ROWS}" --cols "${COLS}" --k "${K}" \
    --outdir "${OUTDIR}" --dtype "${DTYPE}" $( [[ "${STANDARDIZE}" -eq 1 ]] && echo --standardize ) \
    $( [[ "${WHITEN}" -eq 1 ]] && echo --whiten ) --dims "${DIMS}" --rank "${RANK_PLOT}"

  META="${OUTDIR}/metadata.json"
  if [[ -f "${META}" ]]; then
    python - << PYEOF
import json
t=json.load(open("${META}"))["timings_sec"]
print("{np},{load},{center},{cov},{eig},{proj},{tot}".format(
  np=${NP}, load=t["load"], center=t["center"],
  cov=t["cov_allreduce"], eig=t["eig"], proj=t["project"], tot=t["total"]
))
PYEOF
  else
    echo "${NP},NA,NA,NA,NA,NA,NA"
  fi >> "${TIMINGS_CSV}"
done

echo "Done. Timing CSV: ${TIMINGS_CSV}"
