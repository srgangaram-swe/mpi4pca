# MPI4PCA

Parallel principal component analysis using MPI and covariance aggregation.

This project implements PCA for data that is split across MPI ranks. Each rank
loads or generates a local shard of the input matrix, participates in global
reductions for the feature means and covariance/Gram matrix, and can save local
top-k projections after the eigendecomposition.

## Project Layout

- `mpi_pca.py` - main MPI PCA implementation.
- `run_pca.sh` - convenience script that creates a virtual environment,
  installs dependencies, and launches `mpi_pca.py` with `mpirun`.
- `data_vis.py` - plots cumulative variance explained and 2D/3D projections
  from a PCA output directory.
- `iris_data.py` - optional helper for creating small Iris CSV shards.
- `requirements.txt` - Python dependencies.
- `progress_report.pdf` and `Progress Report.docx` - project report files.
- `easley/final_project/` - cluster run scripts, logs, timing CSVs, and sample
  output plots from Easley runs.

## Method

The PCA implementation uses covariance aggregation:

1. Compute the global feature-wise mean with `MPI.Allreduce`.
2. Center each rank's local data shard.
3. Optionally standardize features with a global standard deviation.
4. Compute each rank's local Gram matrix, `X_centered.T @ X_centered`.
5. Sum local Gram matrices with `MPI.Allreduce`.
6. On rank 0, eigendecompose the covariance matrix.
7. Broadcast eigenvalues and eigenvectors to all ranks.
8. Optionally project each local shard onto the top `k` principal components.

This approach communicates `O(d^2)` values for `d` features, making it useful
when the sample count is large and the feature count is moderate.

## Requirements

Install an MPI implementation first:

- macOS with Homebrew: `brew install open-mpi`
- Ubuntu/Debian: `sudo apt-get install -y libopenmpi-dev openmpi-bin`

Python dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

`scikit-learn` is only needed if you want to run `iris_data.py`.

## Quick Start

The easiest way to run a synthetic PCA job is through `run_pca.sh`:

```bash
chmod +x run_pca.sh
./run_pca.sh -n 4 --rows 100000 --cols 64 --k 8 --outdir ./pca_out
```

The script creates `.venv`, installs dependencies, and runs:

```bash
mpirun --oversubscribe --bind-to none -np 4 python mpi_pca.py \
  --rows 100000 --cols 64 --k 8 --outdir ./pca_out --overwrite
```

You can also invoke `mpi_pca.py` directly:

```bash
mpirun -np 4 python mpi_pca.py --rows 100000 --cols 64 --k 8 --outdir ./pca_out
```

## Running On Sharded Data

Input files can be CSV, NPY, or NPZ. CSV files use `--delimiter`, and NPZ files
expect an array named `X` or use the first array in the archive.

```bash
mpirun -np 8 python mpi_pca.py \
  --files "data/shard_*.csv" \
  --delimiter , \
  --k 16 \
  --outdir ./pca_out
```

Files are sorted and assigned to ranks round-robin.

## Useful Options

- `--rows N` - total synthetic rows when `--files` is not provided.
- `--cols D` - synthetic feature count.
- `--k K` - number of principal components to project and save.
- `--dtype float32|float64` - computation and output dtype.
- `--standardize` - z-score features before PCA.
- `--whiten` - scale projected components by inverse square root eigenvalue.
- `--overwrite` - allow reuse of an existing output directory.

## Outputs

When `--outdir` is provided, the run writes:

- `eigenvalues.npy`
- `eigenvectors.npy`
- `mean.npy`
- `metadata.json`
- `projections_rank{r}.npy` for each rank when `--k > 0`

The program also prints max-over-ranks timings for load, centering,
covariance aggregation, eigendecomposition, projection, and total runtime.

## Visualization

After a run with saved outputs, generate plots with:

```bash
python data_vis.py --outdir ./pca_out --rank 0 --dims 2
```

This saves:

- `variance_explained.png`
- `projection_rank0_2d.png` or `projection_rank0_3d.png`

## Easley Runs

The `easley/final_project/` directory contains Slurm scripts, logs, timing CSVs,
and plotted outputs for runs using 1, 2, 4, 8, 16, and 32 ranks. These files are
included as reference artifacts for the final project.
