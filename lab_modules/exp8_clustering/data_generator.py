import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

def generate_cluster_dataset(dataset_name: str, n_samples: int = 300, noise_level: float = 0.05, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    
    if dataset_name == "Blobs (Separable)":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.8, random_state=random_state)
    elif dataset_name == "Overlapping Clusters":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.8, random_state=random_state)
    elif dataset_name == "Concentric Circles":
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=noise_level, random_state=random_state)
    elif dataset_name == "Crescent Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=random_state)
    elif dataset_name == "Anisotropic (Elongated)":
        X_raw, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.6, random_state=random_state)
        transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
        X = np.dot(X_raw, transformation)
    else:
        X, y = make_blobs(n_samples=n_samples, centers=3, random_state=random_state)
        
    df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    df["True_Label"] = y
    return df, X, y

def preprocess_features(X: np.ndarray, apply_scaling: bool = True) -> np.ndarray:
    if apply_scaling:
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    return X.copy()
