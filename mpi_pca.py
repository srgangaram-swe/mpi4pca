#!/usr/bin/env python3
"""
mpi_pca.py — Parallel PCA with MPI (covariance aggregation)

Author: Saif R. Gangaram
Dependencies: mpi4py, numpy

Overview
--------
This script performs Principal Component Analysis (PCA) in parallel using MPI:
each rank holds a shard of the data matrix X (n_local x d). We compute:

1) Global feature-wise mean using Allreduce.
2) Global (uncentered) Gram matrix G = X_centered^T X_centered via Allreduce
   of local contributions.
3) On rank 0: eigen-decomposition of the symmetric covariance matrix
   C = G / (n_total - 1). We use eigh (symmetric).
4) Broadcast eigenvalues/eigenvectors; optionally project local data to
   top-k principal components.

Why covariance aggregation?
---------------------------
- Communication volume is O(d^2) once per stage (Allreduce of dxd matrix),
  independent of the number of samples n_total.
- Good when n_total is very large but d (features) is moderate.
- If d is very large (e.g., > 50k), consider a randomized SVD approach
  with MPI-enabled power iterations (not implemented here, but see notes).

Usage examples
--------------
# 1) Synthetic data, 4 ranks, 1e6 rows (total), 128 columns, keep top 16 PCs
mpirun -np 4 python mpi_pca.py --rows 1000000 --cols 128 --k 16 --outdir ./pca_out

# 2) Load many CSV/NPY shards (each is an n_i x d matrix). We will round-robin
#    assign files across ranks; each rank loads its subset.
mpirun -np 8 python mpi_pca.py --files data/shard_*.csv --delimiter , --k 32 --outdir ./pca_out

# 3) Same with .npy or .npz (expects an array named 'X' in .npz)
mpirun -np 8 python mpi_pca.py --files data/shard_*.npy --k 32 --outdir ./pca_out

Outputs (if --outdir is set)
----------------------------
- outdir/eigenvalues.npy            (shape: [d])
- outdir/eigenvectors.npy           (shape: [d, d]) ordered by descending variance
- outdir/mean.npy                   (shape: [d])
- outdir/projections_rank{r}.npy    (shape: [n_local, k]) for each rank (if --k > 0)
- outdir/metadata.json              (small JSON with sizes/timings)

Notes
-----
- All arrays are processed as float64 by default for numerical robustness.
  You can override with --dtype float32 for memory savings.
- CSV parsing with numpy.loadtxt is simple but not the fastest; consider
  using pre-saved .npy for speed on large runs.
- For very high-dimensional data (d large), consider implementing an MPI
  randomized SVD that only needs matvecs with X and X^T and reduces smaller
  (k x k) matrices. This script provides a clear baseline to extend.

"""

import argparse
import glob
import json
import os
import sys
import time
from typing import List, Tuple, Optional

import numpy as np
from mpi4py import MPI


def log(rank: int, *args, **kwargs):
    """Rank-tagged print that flushes immediately."""
    print(f"[rank {rank}] ", *args, **kwargs, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel PCA with MPI (covariance aggregation)")
    g_data = p.add_argument_group("Data options")
    g_data.add_argument("--files", type=str, default=None,
                        help="Glob for input files (CSV, NPY, or NPZ; NPZ expects array 'X'). "
                             "Files are round-robin assigned to ranks. If omitted, synthetic data is generated.")
    g_data.add_argument("--rows", type=int, default=200000,
                        help="Total number of rows for synthetic data (ignored if --files given).")
    g_data.add_argument("--cols", type=int, default=128,
                        help="Number of columns for synthetic data (ignored if --files given).")
    g_data.add_argument("--rank-chunk", type=int, default=None,
                        help="Override per-rank rows for synthetic data (advanced).")
    g_data.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic data.")
    g_data.add_argument("--delimiter", type=str, default=",", help="CSV delimiter if loading CSV files.")
    g_data.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"],
                        help="Floating dtype for computations and I/O.")
    g_data.add_argument("--standardize", action="store_true",
                        help="After centering, scale columns by global std (z-score) before PCA.")

    g_pca = p.add_argument_group("PCA options")
    g_pca.add_argument("--k", type=int, default=0,
                       help="If > 0, compute local projections to top-k PCs and save per-rank outputs. "
                            "Else, only eigenpairs are computed.")
    g_pca.add_argument("--whiten", action="store_true",
                       help="Scale principal components by 1/sqrt(eigenvalue) when projecting (PCA whiten).")

    g_io = p.add_argument_group("I/O options")
    g_io.add_argument("--outdir", type=str, default=None, help="Directory to write outputs.")
    g_io.add_argument("--overwrite", action="store_true", help="Allow overwriting outdir if it exists.")

    g_perf = p.add_argument_group("Performance")
    g_perf.add_argument("--blockrows", type=int, default=50000,
                        help="CSV load blockrows hint (best-effort; currently loadtxt loads whole file).")
    g_perf.add_argument("--dry-run", action="store_true",
                        help="Run setup and timing without saving outputs (still computes PCA).")

    args = p.parse_args()
    return args


def assign_files_to_rank(files_glob: str, comm: MPI.Comm) -> List[str]:
    """Round-robin assign sorted file list to ranks."""
    files = sorted(glob.glob(files_glob))
    if not files:
        raise FileNotFoundError(f"No files matched glob: {files_glob}")
    my_files = [f for i, f in enumerate(files) if i % comm.Get_size() == comm.Get_rank()]
    return my_files


def load_one(path: str, delimiter: str, dtype: np.dtype, rank: int) -> np.ndarray:
    """Load a single file as a 2D array (n_i x d). Supports .csv/.txt, .npy, .npz (array name 'X')."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        log(rank, f"Loading CSV: {path}")
        arr = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
    elif ext == ".npy":
        log(rank, f"Loading NPY: {path}")
        arr = np.load(path).astype(dtype, copy=False)
    elif ext == ".npz":
        log(rank, f"Loading NPZ: {path}")
        with np.load(path) as z:
            key = "X" if "X" in z.files else z.files[0]
            arr = z[key].astype(dtype, copy=False)
    else:
        raise ValueError(f"Unsupported file extension: {ext} for {path}")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def load_data(files_glob: Optional[str],
              delimiter: str,
              dtype: np.dtype,
              total_rows: int,
              cols: int,
              rank_chunk: Optional[int],
              seed: int,
              comm: MPI.Comm) -> Tuple[np.ndarray, int, int]:
    """
    Load or synthesize the local data for this rank.

    Returns:
        X_local: (n_local x d) array
        n_total: total rows across all ranks
        d: number of columns (features)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if files_glob:
        my_files = assign_files_to_rank(files_glob, comm)
        if not my_files:
            log(rank, "No files assigned to this rank. Creating empty shard.")
            X_local = np.zeros((0, 0), dtype=dtype)
        else:
            shards = [load_one(p, delimiter, dtype, rank) for p in my_files]
            # Validate consistent column counts
            d0 = shards[0].shape[1]
            for s in shards:
                if s.shape[1] != d0:
                    raise ValueError(f"Column mismatch among assigned files for rank {rank}: got {s.shape[1]} vs {d0}")
            X_local = np.vstack(shards) if shards else np.zeros((0, d0), dtype=dtype)

        # Gather shapes to compute n_total and d
        n_local = np.array([X_local.shape[0]], dtype=np.int64)
        d_local = np.array([X_local.shape[1] if X_local.size else 0], dtype=np.int64)
        n_total = np.array([0], dtype=np.int64)
        d_max = np.array([0], dtype=np.int64)
        comm.Allreduce(n_local, n_total, op=MPI.SUM)
        comm.Allreduce(d_local, d_max, op=MPI.MAX)
        d = int(d_max[0])
        # If this rank had 0 columns, fix shape to (0, d) for downstream ops
        if X_local.size == 0 and d > 0:
            X_local = np.zeros((0, d), dtype=dtype)
        return X_local, int(n_total[0]), d

    # Synthetic path
    # Partition rows roughly evenly; allow override of per-rank rows
    if rank_chunk is not None:
        n_local = rank_chunk
        n_total = n_local * size
    else:
        q, r = divmod(total_rows, size)
        n_local = q + (1 if rank < r else 0)
        n_total = total_rows

    rng = np.random.default_rng(seed + rank)
    # Create a low-rank-ish dataset: mix a few latent factors + noise
    k_latent = max(4, min(32, cols // 4))
    W = rng.normal(0, 1.0, size=(k_latent, cols)).astype(dtype)
    Z = rng.normal(0, 1.0, size=(n_local, k_latent)).astype(dtype)
    noise = rng.normal(0, 0.1, size=(n_local, cols)).astype(dtype)
    X_local = Z @ W + noise

    return X_local, n_total, cols


def global_mean_std(X_local: np.ndarray, comm: MPI.Comm) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute global mean (and std if requested later) across rows via Allreduce of sums and counts."""
    rank = comm.Get_rank()
    n_local = X_local.shape[0]
    d = X_local.shape[1] if X_local.ndim == 2 and X_local.size > 0 else 0

    # Handle empty shard
    if d == 0:
        # Determine max d across ranks to shape buffers
        d_max = np.array([0], dtype=np.int64)
        comm.Allreduce(np.array([0], dtype=np.int64), d_max, op=MPI.MAX)
        d = int(d_max[0])
        local_sum = np.zeros(d, dtype=X_local.dtype)
    else:
        local_sum = X_local.sum(axis=0)

    global_sum = np.zeros_like(local_sum)
    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    n_total = np.array([0], dtype=np.int64)
    comm.Allreduce(np.array([n_local], dtype=np.int64), n_total, op=MPI.SUM)

    if n_total[0] == 0:
        raise ValueError("Total number of rows across ranks is zero. No data to process.")

    mean = global_sum / n_total[0]
    return mean, None  # std deferred to separate function to avoid extra pass


def global_std(X_local_centered: np.ndarray, comm: MPI.Comm) -> np.ndarray:
    """Compute global std from centered data via Allreduce of sum of squares."""
    # sum of squares across all rows for each feature
    ss_local = np.sum(X_local_centered * X_local_centered, axis=0)
    ss_global = np.zeros_like(ss_local)
    comm.Allreduce(ss_local, ss_global, op=MPI.SUM)
    n_total = np.array([0], dtype=np.int64)
    comm.Allreduce(np.array([X_local_centered.shape[0]], dtype=np.int64), n_total, op=MPI.SUM)
    # unbiased estimator uses (n_total - 1), but for standardization we prefer population std
    var = ss_global / max(1, int(n_total[0]))
    std = np.sqrt(np.maximum(var, 1e-30))
    return std


def cov_allreduce(X_local_centered: np.ndarray, comm: MPI.Comm) -> Tuple[np.ndarray, int]:
    """Compute global Gram matrix G = X^T X via Allreduce of local contributions."""
    n_local, d = X_local_centered.shape
    # Local contribution (d x d)
    # Compute with BLAS-friendly order: X^T @ X
    G_local = X_local_centered.T @ X_local_centered  # (d x d)
    G_global = np.zeros_like(G_local)
    comm.Allreduce(G_local, G_global, op=MPI.SUM)

    n_total = np.array([0], dtype=np.int64)
    comm.Allreduce(np.array([n_local], dtype=np.int64), n_total, op=MPI.SUM)
    return G_global, int(n_total[0])


def eig_from_cov(G: np.ndarray, n_total: int, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenpairs of covariance matrix (G / (n_total - 1)), sorted desc by eigenvalue."""
    if n_total <= 1:
        raise ValueError("Need at least 2 total rows to form covariance.")
    C = G / (n_total - 1)  # unbiased covariance
    # eigh for symmetric PSD matrix
    evals, evecs = np.linalg.eigh(C)
    # ascending -> descending
    order = np.argsort(evals)[::-1]
    evals = evals[order].astype(dtype, copy=False)
    evecs = evecs[:, order].astype(dtype, copy=False)
    return evals, evecs


def project_local(Xc_local: np.ndarray,
                  evecs: np.ndarray,
                  evals: np.ndarray,
                  k: int,
                  whiten: bool,
                  dtype: np.dtype) -> np.ndarray:
    """Project centered (or standardized) local data to top-k components."""
    U = evecs[:, :k]
    if whiten:
        scales = 1.0 / np.sqrt(np.maximum(evals[:k], 1e-30))
        U = U * scales  # whitened PCs: components scaled
    Z = Xc_local @ U
    return Z.astype(dtype, copy=False)


def maybe_make_outdir(outdir: Optional[str], overwrite: bool, rank: int):
    if outdir is None:
        return
    if os.path.exists(outdir):
        if not overwrite and rank == 0:
            # Only rank 0 raises the error, but all ranks will exit once it happens
            raise FileExistsError(
                f"Output directory '{outdir}' exists. Use --overwrite to allow re-use."
            )
    else:
        # All ranks attempt to create; exist_ok=True avoids races
        os.makedirs(outdir, exist_ok=True)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64

    t0 = time.time()

    # Load or synthesize data
    X_local, n_total, d = load_data(
        files_glob=args.files,
        delimiter=args.delimiter,
        dtype=dtype,
        total_rows=args.rows,
        cols=args.cols,
        rank_chunk=args.rank_chunk,
        seed=args.seed,
        comm=comm,
    )
    t_load = time.time()

    if X_local.size == 0 and d == 0:
        if rank == 0:
            print("No data loaded or generated. Exiting.", flush=True)
        MPI.Finalize()
        return

    # Global mean
    mean, _ = global_mean_std(X_local, comm)
    # Center
    if X_local.size:
        Xc_local = X_local - mean
    else:
        Xc_local = np.zeros((0, d), dtype=dtype)

    # Optional standardization (after centering)
    if args.standardize:
        std = global_std(Xc_local, comm)
        if Xc_local.size:
            Xc_local /= std
    else:
        std = None

    t_center = time.time()

    # Allreduce covariance (Gram matrix)
    G, n_total_check = cov_allreduce(Xc_local, comm)
    assert n_total_check == n_total, "n_total mismatch in covariance stage"
    t_cov = time.time()

    # Eigendecomposition on root
    if rank == 0:
        evals, evecs = eig_from_cov(G, n_total, dtype)
    else:
        evals = np.empty(d, dtype=dtype)
        evecs = np.empty((d, d), dtype=dtype)

    # Broadcast eigenpairs
    comm.Bcast(evals, root=0)
    comm.Bcast(evecs, root=0)
    t_eig = time.time()

    # Optional projection to top-k
    if args.k and args.k > 0:
        k = min(args.k, d)
        Z_local = project_local(Xc_local, evecs, evals, k, args.whiten, dtype)
    else:
        Z_local = None

    t_proj = time.time()

    # Optional output
    if args.outdir:
        maybe_make_outdir(args.outdir, args.overwrite, rank)
        # Ensure directory exists before any rank writes
        comm.Barrier()

        if rank == 0:
            np.save(os.path.join(args.outdir, "eigenvalues.npy"), evals)
            np.save(os.path.join(args.outdir, "eigenvectors.npy"), evecs)
            np.save(os.path.join(args.outdir, "mean.npy"), mean)
            meta = {
                "n_total": int(n_total),
                "d": int(d),
                "k": int(min(args.k, d)) if args.k else 0,
                "dtype": args.dtype,
                "standardize": bool(args.standardize),
                "whiten": bool(args.whiten),
                "ranks": int(size),
                "timings_sec": {
                    "load": t_load - t0,
                    "center": t_center - t_load,
                    "cov_allreduce": t_cov - t_center,
                    "eig": t_eig - t_cov,
                    "project": t_proj - t_eig,
                    "total": t_proj - t0,
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
            with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

        if Z_local is not None:
            np.save(os.path.join(args.outdir, f"projections_rank{rank}.npy"), Z_local)

    # Final logging
    # Gather timings for a quick report
    local_times = np.array(
        [t_load - t0, t_center - t_load, t_cov - t_center, t_eig - t_cov, t_proj - t_eig, t_proj - t0],
        dtype=np.float64,
    )
    times = np.zeros_like(local_times)
    comm.Reduce(local_times, times, op=MPI.MAX, root=0)

    if rank == 0:
        stages = ["load", "center", "cov_allreduce", "eig", "project", "total"]
        print("\n=== PCA Timing (max over ranks, seconds) ===", flush=True)
        for s, val in zip(stages, times):
            print(f"{s:>14}: {val:8.3f}", flush=True)
        print(f"n_total={n_total}, d={d}, ranks={size}, k={args.k}", flush=True)

    # Optionally, verify orthonormality of eigenvectors (on root)
    if rank == 0:
        ortho_err = np.linalg.norm(evecs.T @ evecs - np.eye(d, dtype=dtype))
        print(f"Orthonormality check ||V^T V - I||_F = {ortho_err:.3e}", flush=True)

    MPI.Finalize()


if __name__ == "__main__":
    main()
