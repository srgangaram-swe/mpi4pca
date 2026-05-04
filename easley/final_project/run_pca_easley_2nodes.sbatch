#!/bin/bash
#SBATCH -J pca-mpi-64
#SBATCH -o pca-mpi-64-%j.out
#SBATCH -e pca-mpi-64-%j.err
#SBATCH -t 00:30:00
#SBATCH -p h100
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive

module purge
module load python/3.10 2>/dev/null || true
module load openmpi 2>/dev/null || module load mpi 2>/dev/null || true
# module load mpi4py matplotlib  # if available as modules

export VENVDIR="${VENVDIR:-.venv}"
export NP_LIST="${NP_LIST:-64}"   # exactly 64
export ROWS="${ROWS:-10000000}"
export COLS="${COLS:-256}"
export K="${K:-16}"
export OUTBASE="${OUTBASE:-pca_out}"
export DTYPE="${DTYPE:-float64}"
export STANDARDIZE="${STANDARDIZE:-0}"
export WHITEN="${WHITEN:-0}"
export DIMS="${DIMS:-2}"
export RANK_PLOT="${RANK_PLOT:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

if [[ ! -x ./run_pca_and_vis.sh ]]; then chmod +x run_pca_and_vis.sh; fi
echo "np,load,center,cov_allreduce,eig,project,total" > "timings_${SLURM_JOB_ID}.csv"

NP=64
OUTDIR="${OUTBASE}_np${NP}"
./run_pca_and_vis.sh -n "${NP}" --rows "${ROWS}" --cols "${COLS}" --k "${K}" \
  --outdir "${OUTDIR}" --dtype "${DTYPE}" $( [[ "${STANDARDIZE}" -eq 1 ]] && echo --standardize ) \
  $( [[ "${WHITEN}" -eq 1 ]] && echo --whiten ) --dims "${DIMS}" --rank "${RANK_PLOT}"

META="${OUTDIR}/metadata.json"
if [[ -f "${META}" ]]; then
  python - << PYEOF >> "timings_${SLURM_JOB_ID}.csv"
import json
t=json.load(open("${META}"))["timings_sec"]
print("{np},{load},{center},{cov},{eig},{proj},{tot}".format(
  np=${NP}, load=t["load"], center=t["center"],
  cov=t["cov_allreduce"], eig=t["eig"], proj=t["project"], tot=t["total"]
))
PYEOF
else
  echo "${NP},NA,NA,NA,NA,NA,NA" >> "timings_${SLURM_JOB_ID}.csv"
fi
