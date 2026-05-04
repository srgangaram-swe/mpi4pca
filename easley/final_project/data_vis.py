#!/usr/bin/env python3
"""
data_vis.py — Plot results from mpi_pca.py output directory.

Usage:
    python data_vis.py --outdir ./pca_test [--rank 0] [--dims 3]

It will:
  - Plot eigenvalue spectrum (variance explained)
  - Plot 2D or 3D projection scatter (if projections_rank*.npy exist)
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_metadata(outdir):
    meta_path = os.path.join(outdir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return meta

def plot_eigen_spectrum(outdir, evals):
    plt.figure()
    total = np.sum(evals)
    explained = np.cumsum(evals) / total
    plt.plot(np.arange(1, len(evals)+1), explained, marker='o')
    plt.title("Cumulative Variance Explained")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "variance_explained.png"))
    plt.show()

def load_projection(outdir, rank=0):
    path = os.path.join(outdir, f"projections_rank{rank}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No projection file found for rank {rank}: {path}")
    return np.load(path)

def plot_projection(Z, dims, outdir, rank):
    if dims == 2:
        plt.figure()
        plt.scatter(Z[:,0], Z[:,1], s=15, alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA Projection (Rank {rank})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"projection_rank{rank}_2d.png"))
        plt.show()
    elif dims == 3 and Z.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Z[:,0], Z[:,1], Z[:,2], s=15, alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.title(f"PCA Projection (Rank {rank})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"projection_rank{rank}_3d.png"))
        plt.show()
    else:
        print(f"Projection only has {Z.shape[1]} dims; skipping {dims}D plot.")

def main():
    parser = argparse.ArgumentParser(description="Visualize PCA output directory")
    parser.add_argument("--outdir", required=True, help="Directory containing PCA outputs")
    parser.add_argument("--rank", type=int, default=0, help="Which rank's projection to plot (default=0)")
    parser.add_argument("--dims", type=int, default=2, choices=[2,3], help="Plot 2D or 3D projection")
    args = parser.parse_args()

    evals_path = os.path.join(args.outdir, "eigenvalues.npy")
    if not os.path.exists(evals_path):
        raise FileNotFoundError(f"Missing eigenvalues.npy in {args.outdir}")

    evals = np.load(evals_path)
    meta = load_metadata(args.outdir)
    print("Loaded metadata:", json.dumps(meta, indent=2))
    print("Top eigenvalues:", evals[:5])

    plot_eigen_spectrum(args.outdir, evals)

    # Try to load projection file (if exists)
    proj_files = glob.glob(os.path.join(args.outdir, "projections_rank*.npy"))
    if proj_files:
        Z = load_projection(args.outdir, rank=args.rank)
        plot_projection(Z, args.dims, args.outdir, args.rank)
    else:
        print("No projection files found — did you run with --k > 0?")

if __name__ == "__main__":
    main()