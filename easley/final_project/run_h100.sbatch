#!/bin/bash
#
# run_h100_fixed.sbatch — one-node h100 speedup demo (no user-site confusion)
#
#SBATCH -J pca-speedup
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=32     # ranks to sweep up to
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o speedup-%j.out
#SBATCH -e speedup-%j.err

set -euo pipefail

########## Modules (match Easley) ##########
module purge
module load python/3.13.2-4wqv
module load openmpi/4.1.7-3ilj

########## Build a clean venv bound to this Open MPI ##########
python -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip wheel numpy matplotlib
# Compile mpi4py against the loaded Open MPI:
MPICC="$(which mpicc)" python -m pip install -q --no-binary=mpi4py mpi4py==4.1.1

# Keep Python from seeing ~/.local (old/bad mpi4py)
export PYTHONNOUSERSITE=1

########## CPU-only + no BLAS oversubscription ##########
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

########## Preflight (fail fast if mpi4py/MPI is wrong) ##########
python - <<'PY'
from mpi4py import MPI
print("mpi4py OK:", MPI.Get_library_version())
PY

########## Problem + sweep settings (override via --export=ALL,VAR=...) ##########
ROWS=${ROWS:-3000000}          # total rows
COLS=${COLS:-256}              # features
K=${K:-16}                     # projected dims
DTYPE=${DTYPE:-float64}
OUTBASE=${OUTBASE:-pca_out}
NP_LIST=${NP_LIST:-"1 2 4 8 16 32"}

########## Sanity: need your scripts ##########
test -f mpi_pca.py
test -f data_vis.py || true   # plotting is optional; we won't fail if missing

CSV="timings_${SLURM_JOB_ID}.csv"
echo "np,total" > "$CSV"

for NP in $NP_LIST; do
  OUTDIR="${OUTBASE}_np${NP}"
  echo "=== NP=${NP} ==="

  # Run the PCA (MPI ranks bound to cores)
  srun -n "${NP}" --cpu-bind=cores python -m mpi4py mpi_pca.py \
      --rows "${ROWS}" --cols "${COLS}" --k "${K}" --dtype "${DTYPE}" \
      --outdir "${OUTDIR}" --overwrite

  # Optional: quick plot (won't fail job if missing/matplotlib issue)
  if [[ -f data_vis.py ]]; then
    python data_vis.py --outdir "${OUTDIR}" --rank 0 --dims 2 || true
  fi

  META="${OUTDIR}/metadata.json"
  if [[ -f "$META" ]]; then
    python - <<PY >> "$CSV"
import json
t=json.load(open("${META}"))["timings_sec"]
print(f"{${NP}},{t['total']}")
PY
  else
    echo "WARNING: missing ${META}" >&2
    echo "${NP},NA" >> "$CSV"
  fi
done

# Print a tiny speedup table (relative to NP=1)
python - << 'PY' "$CSV"
import csv, sys
rows=[]
with open(sys.argv[1]) as f:
    rd=csv.DictReader(f)
    for r in rd:
        try: rows.append((int(r["np"]), float(r["total"])))
        except: pass
rows=sorted(rows)
base=dict(rows).get(1, None)
print("\nSpeedup (vs NP=1):")
print("{:>6} {:>10} {:>10}".format("np","time(s)","speedup"))
for np,t in rows:
    if base and t>0:
        print("{:>6} {:>10.3f} {:>10.2f}".format(np,t, base/t))
    else:
        print("{:>6} {:>10} {:>10}".format(np,"NA","NA"))
PY
