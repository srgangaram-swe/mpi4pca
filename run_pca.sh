#!/usr/bin/env bash
#
# run_pca.sh — setup venv, install deps, and run MPI-parallel PCA
#
# This script will:
#   1) Check for python3 and mpirun.
#   2) Create a Python virtual environment (if it doesn't exist).
#   3) Install numpy + mpi4py (or use requirements.txt if present).
#   4) Launch mpi_pca.py under mpirun with --oversubscribe.
#
# Usage examples:
#   chmod +x run_pca.sh
#
#   # Synthetic data, 8 ranks
#   ./run_pca.sh -n 8 --rows 1000000 --cols 128 --k 16 --outdir ./pca_out
#
#   # CSV shards
#   ./run_pca.sh -n 8 --files "data/shard_*.csv" --k 32 --outdir ./pca_out
#
set -euo pipefail

# -------- Defaults (can be overridden via env vars or flags) --------
VENVDIR="${VENVDIR:-.venv}"   # directory where venv will live
SCRIPT="${SCRIPT:-mpi_pca.py}"

NP="${NP:-8}"                 # number of MPI ranks
ROWS="${ROWS:-1000000}"       # synthetic rows if --files not used
COLS="${COLS:-128}"           # synthetic cols if --files not used
K="${K:-16}"                  # PCs to project
OUTDIR="${OUTDIR:-./pca_out}" # output directory

FILES_GLOB="${FILES_GLOB:-}"  # glob for CSV/NPY/NPZ shards
DTYPE="${DTYPE:-float64}"     # float32 or float64
STANDARDIZE="${STANDARDIZE:-0}"
WHITEN="${WHITEN:-0}"

usage() {
  cat <<EOF
Usage: $0 [-n NP] [--rows N] [--cols D] [--k K] [--outdir DIR] [--files GLOB]
          [--dtype float32|float64] [--standardize] [--whiten]

Environment overrides:
  VENVDIR   Path for venv (default: .venv)
  SCRIPT    PCA script path (default: mpi_pca.py)
  NP        MPI ranks (default: 8)
  ROWS      Synthetic total rows (default: 1000000)
  COLS      Synthetic cols (default: 128)
  K         Number of PCs to project (default: 16)
  OUTDIR    Output directory (default: ./pca_out)

Examples:
  # Synthetic data, 8 ranks
  ./run_pca.sh -n 8 --rows 1000000 --cols 128 --k 16 --outdir ./pca_out

  # CSV shards
  ./run_pca.sh -n 8 --files "data/shard_*.csv" --k 32 --outdir ./pca_out
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
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1;;
  esac
done

# -------- Check prerequisites: python3 and mpirun --------
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH." >&2
  echo "Install Python 3 (e.g., on Ubuntu: sudo apt-get install -y python3 python3-venv python3-pip)" >&2
  exit 1
fi

if ! command -v mpirun >/dev/null 2>&1; then
  echo "ERROR: mpirun not found in PATH." >&2
  echo "Install an MPI implementation first (Open MPI or MPICH)." >&2
  echo "  Ubuntu (Open MPI): sudo apt-get install -y libopenmpi-dev openmpi-bin" >&2
  echo "  macOS (Homebrew):  brew install open-mpi" >&2
  exit 1
fi

# -------- Create & activate virtual environment --------
if [[ ! -d "${VENVDIR}" ]]; then
  echo "Creating virtual environment in '${VENVDIR}' ..."
  python3 -m venv "${VENVDIR}"
fi

# shellcheck disable=SC1090
source "${VENVDIR}/bin/activate"

# Now 'python' and 'pip' refer to the venv's interpreter
python -m pip install --upgrade pip wheel

# -------- Install Python dependencies --------
REQ_FILE="requirements.txt"
if [[ -f "${REQ_FILE}" ]]; then
  echo "Installing dependencies from ${REQ_FILE} ..."
  pip install -r "${REQ_FILE}"
else
  echo "Installing minimal dependencies (numpy, mpi4py) ..."
  pip install numpy mpi4py
fi

# Quick sanity check
python - << 'EOF'
import sys
print("Python executable:", sys.executable)
import numpy as np
from mpi4py import MPI
print("numpy version:", np.__version__)
print("mpi4py version:", MPI.Get_version())
EOF

# -------- Verify PCA script exists --------
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: PCA script '${SCRIPT}' not found in directory: $(pwd)" >&2
  echo "Make sure mpi_pca.py is next to run_pca.sh or set SCRIPT=/path/to/mpi_pca.py" >&2
  exit 1
fi

# -------- Build argument list for mpi_pca.py --------
PYARGS=( "--dtype" "${DTYPE}" "--k" "${K}" "--outdir" "${OUTDIR}" "--overwrite" )

if [[ -n "${FILES_GLOB}" ]]; then
  PYARGS+=( "--files" "${FILES_GLOB}" )
else
  PYARGS+=( "--rows" "${ROWS}" "--cols" "${COLS}" )
fi

if [[ "${STANDARDIZE}" -eq 1 ]]; then
  PYARGS+=( "--standardize" )
fi
if [[ "${WHITEN}" -eq 1 ]]; then
  PYARGS+=( "--whiten" )
fi

# -------- Run under mpirun with oversubscribe --------
echo "Running with MPI:"
echo "  ranks     = ${NP}"
echo "  script    = ${SCRIPT}"
echo "  outdir    = ${OUTDIR}"
echo "  files     = ${FILES_GLOB:-<synthetic>}"

echo
echo "Command:"
echo "mpirun --oversubscribe --bind-to none -np ${NP} python ${SCRIPT} ${PYARGS[*]}"
echo

mpirun --oversubscribe --bind-to none -np "${NP}" python "${SCRIPT}" "${PYARGS[@]}"
