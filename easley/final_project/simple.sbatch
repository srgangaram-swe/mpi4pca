#!/bin/bash
#
# run_h100_super_simple.sbatch — zero-frills demo speedup on one h100 node
#
#SBATCH -J pca-speedup-lite
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=32      # <= CPU cores on the node
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o speedup-lite-%j.out
#SBATCH -e speedup-lite-%j.err

set -euo pipefail

module purge
# Adjust these to Easley module names:
module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
module load openmpi 2>/dev/null || module load mpi 2>/dev/null || true
module load mpi4py 2>/dev/null || true
module load matplotlib 2>/dev/null || true
module list || true

# CPU-only
export CUDA_VISIBLE_DEVICES=""
# No BLAS oversubscription
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

# Problem size (quick but not tiny)
ROWS=${ROWS:-3000000}
COLS=${COLS:-256}
K=${K:-16}
DTYPE=${DTYPE:-float64}
OUTBASE=${OUTBASE:-pca_out}
NP_LIST=${NP_LIST:-"1 2 4 8 16 32"}

# Make sure scripts exist
test -f mpi_pca.py
test -f data_vis.py

CSV="timings_${SLURM_JOB_ID}.csv"
echo "np,total" > "$CSV"

for NP in $NP_LIST; do
  OUTDIR="${OUTBASE}_np${NP}"
  echo "=== NP=${NP} ==="

  # Run the PCA (no venv, use site Python + mpi4py). If cluster requires MPI module:
  srun -n "${NP}" --cpu-bind=cores python -m mpi4py mpi_pca.py \
      --rows "${ROWS}" --cols "${COLS}" --k "${K}" --dtype "${DTYPE}" --outdir "${OUTDIR}" --overwrite

  # Visualize (single-process)
  python data_vis.py --outdir "${OUTDIR}" --rank 0 --dims 2 || true

  META="${OUTDIR}/metadata.json"
  if [[ -f "$META" ]]; then
    python - << PYEOF >> "$CSV"
import json
t=json.load(open("${META}"))["timings_sec"]
print(f"{${{NP}}},{t['total']}")
PYEOF
  else
    echo "WARNING: missing ${META}" >&2
    echo "${NP},NA" >> "$CSV"
  fi
done

# Print speedup table
python - << 'PYEOF' "$CSV"
import csv, sys
rows=[]
with open(sys.argv[1]) as f:
    rd=csv.DictReader(f)
    for r in rd:
        try:
            rows.append((int(r["np"]), float(r["total"])))
        except:
            pass
rows=sorted(rows)
base=dict(rows).get(1, None)
print("\nSpeedup table (relative to NP=1):")
print("{:>6} {:>10} {:>10}".format("np","time(s)","speedup"))
for np,t in rows:
    if base and t>0:
        print("{:>6} {:>10.3f} {:>10.2f}".format(np,t, base/t))
    else:
        print("{:>6} {:>10} {:>10}".format(np,"NA","NA"))
PYEOF

echo "Done. Timings in: $CSV"
