#!/bin/bash
#
# run_h100_simple.sbatch — minimal 1-node scaling sweep to demonstrate speedup
#
#SBATCH -J pca-speedup
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=32      # up to 32 ranks
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -o speedup-%j.out
#SBATCH -e speedup-%j.err

module purge
module load python/3.10 2>/dev/null || true
module load openmpi 2>/dev/null || module load mpi 2>/dev/null || true

# Keep it CPU-only even on h100
export CUDA_VISIBLE_DEVICES=""

# Keep BLAS from spawning threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

# Simple sizes that finish quickly but show parallelism
ROWS=${ROWS:-3000000}   # total rows (3M)
COLS=${COLS:-256}       # features
K=${K:-16}              # project dims
DTYPE=${DTYPE:-float64}
OUTBASE=${OUTBASE:-pca_out}
NP_LIST=${NP_LIST:-"1 2 4 8 16 32"}

# Ensure runner exists
if [[ ! -x ./run_pca_and_vis.sh ]]; then
  echo "ERROR: run_pca_and_vis.sh is missing or not executable"; exit 1
fi

# CSV header
CSV="timings_${SLURM_JOB_ID}.csv"
echo "np,total" > "$CSV"

for NP in $NP_LIST; do
  OUTDIR="${OUTBASE}_np${NP}"
  echo "=== Running NP=${NP} ==="
  ./run_pca_and_vis.sh -n "${NP}" --rows "${ROWS}" --cols "${COLS}" --k "${K}" --outdir "${OUTDIR}" --dtype "${DTYPE}"

  META="${OUTDIR}/metadata.json"
  if [[ -f "$META" ]]; then
    python - << PYEOF >> "$CSV"
import json
t=json.load(open("${META}"))["timings_sec"]
print(f"{${{NP}}},{t['total']}")
PYEOF
  else
    echo "${NP},NA" >> "$CSV"
  fi
done

# Print a tiny speedup table relative to NP=1
python - << 'PYEOF'
import csv, sys
rows=[]
with open([x for x in sys.argv if x.endswith(".csv")][0]) as f:
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
PYEOF "$CSV"

echo
echo "Done. Timings in: $CSV"
