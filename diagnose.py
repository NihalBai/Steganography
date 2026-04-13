import numpy as np

X = np.load("features.npy")
y = np.load("labels.npy")

print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"y unique values : {np.unique(y, return_counts=True)}")
print(f"X min/max : {X.min():.3f} / {X.max():.3f}")
print(f"X sample (first 5 values): {X[0][:5]}")