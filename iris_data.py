"""
iris_data.py

Get iris small dataset as CSV shards
to test on PCA model

"""

from sklearn.datasets import load_iris
import numpy as np

if __name__ == "__main__":
    X = load_iris().data
    for i in range(4):
        np.savetxt(f"iris/shard_{i}.csv", X[i*38:(i+1)*38], delimiter=",")
