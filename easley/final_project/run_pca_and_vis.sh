#!/usr/bin/env bash
set -euo pipefail

# run_pca_and_vis.sh — run MPI PCA then generate plots (cluster-friendly)
# Usage (inside an allocated node or via sbatch):
#   ./run_pca_and_vis.sh -n 32 --rows 5000000 --cols 256 --k 16 --outdir ./pca_out_32
#
# Expects mpi_pca.py and data_vis.py in the current directory.
# Sets a non-interactive matplotlib backend and pins BLAS threads to 1.

# -------- Defaults (override via flags/env) --------
VENVDIR="${VENVDIR:-.venv}"
SCRIPT="${SCRIPT:-mpi_pca.py}"
VIS_SCRIPT="${VIS_SCRIPT:-data_vis.py}"
NP="${NP:-8}"
ROWS="${ROWS:-1000000}"
COLS="${COLS:-128}"
K="${K:-16}"
OUTDIR="${OUTDIR:-./pca_out}"
FILES_GLOB="${FILES_GLOB:-}"
DTYPE="${DTYPE:-float64}"
STANDARDIZE="${STANDARDIZE:-0}"
WHITEN="${WHITEN:-0}"
DIMS="${DIMS:-2}"     # visualization dims (2 or 3)
RANK_PLOT="${RANK_PLOT:-0}"  # which rank's projection to plot

usage() {
  cat <<EOF
Usage: $0 [-n NP] [--rows N] [--cols D] [--k K] [--outdir DIR] [--files GLOB]
          [--dtype float32|float64] [--standardize] [--whiten] [--dims 2|3] [--rank R]
EOF
}

# -------- Parse CLI flags --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--np) NP="$2"; shift 2;;
    --rows) ROWS="$2"; shift 2;;
    --cols) COLS="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --files) FILES_GLOB="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --standardize) STANDARDIZE=1; shift;;
    --whiten) WHITEN=1; shift;;
    --dims) DIMS="$2"; shift 2;;
    --rank) RANK_PLOT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done

# -------- Environment for clusters --------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"  # headless plotting

# -------- Python venv --------
if [[ ! -d "${VENVDIR}" ]]; then
  python3 -m venv "${VENVDIR}"
fi
# shellcheck disable=SC1090
source "${VENVDIR}/bin/activate"
python -m pip install --upgrade pip wheel
# Prefer requirements.txt if present (so matplotlib for data_vis is installed)
if [[ -f "requirements.txt" ]]; then
  pip install -r requirements.txt
else
  pip install numpy mpi4py matplotlib
fi

# -------- Sanity checks --------
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: ${SCRIPT} not found in $(pwd)"; exit 1;
fi
if [[ ! -f "${VIS_SCRIPT}" ]]; then
  echo "ERROR: ${VIS_SCRIPT} not found in $(pwd)"; exit 1;
fi

# -------- Build PCA args --------
PYARGS=( "--dtype" "${DTYPE}" "--k" "${K}" "--outdir" "${OUTDIR}" "--overwrite" )
if [[ -n "${FILES_GLOB}" ]]; then
  PYARGS+=( "--files" "${FILES_GLOB}" )
else
  PYARGS+=( "--rows" "${ROWS}" "--cols" "${COLS}" )
fi
if [[ "${STANDARDIZE}" -eq 1 ]]; then PYARGS+=( "--standardize" ); fi
if [[ "${WHITEN}" -eq 1 ]]; then PYARGS+=( "--whiten" ); fi

# -------- Launch MPI job --------
if command -v srun >/dev/null 2>&1; then
  echo "Launching with srun (ntasks=${NP}) ..."
  srun -n "${NP}" --cpu-bind=cores python "${SCRIPT}" "${PYARGS[@]}"
else
  echo "Launching with mpirun (np=${NP}) ..."
  mpirun -np "${NP}" python "${SCRIPT}" "${PYARGS[@]}"
fi

# -------- Visualization --------
echo "Generating visualizations from ${OUTDIR} ..."
python "${VIS_SCRIPT}" --outdir "${OUTDIR}" --rank "${RANK_PLOT}" --dims "${DIMS}"

echo "Done. Outputs in: ${OUTDIR}"
