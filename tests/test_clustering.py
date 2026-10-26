import pytest
import numpy as np
from lab_modules.exp8_clustering.data_generator import generate_cluster_dataset, preprocess_features
from lab_modules.exp8_clustering.models import run_kmeans, run_gmm, run_dbscan, compute_silhouette

def test_deterministic_dbscan_reproducibility():
    """Verify DBSCAN on deterministic Crescent Moons produces identical Silhouette scores across re-runs."""
    _, X_raw1, _ = generate_cluster_dataset("Crescent Moons", n_samples=300, noise_level=0.05, random_state=42)
    X1 = preprocess_features(X_raw1, apply_scaling=True)
    res1 = run_dbscan(X1, eps=0.4, min_samples=5)
    
    _, X_raw2, _ = generate_cluster_dataset("Crescent Moons", n_samples=300, noise_level=0.05, random_state=42)
    X2 = preprocess_features(X_raw2, apply_scaling=True)
    res2 = run_dbscan(X2, eps=0.4, min_samples=5)
    
    assert res1["silhouette"] == res2["silhouette"]
    assert np.array_equal(res1["labels"], res2["labels"])

def test_gmm_seeded_reproducibility():
    """Verify GMM with explicit random_state produces deterministic clusters."""
    _, X_raw, _ = generate_cluster_dataset("Crescent Moons", n_samples=300, random_state=42)
    X = preprocess_features(X_raw, apply_scaling=True)
    
    res1 = run_gmm(X, n_components=3, random_state=42)
    res2 = run_gmm(X, n_components=3, random_state=42)
    
    assert res1["silhouette"] == res2["silhouette"]
    assert np.array_equal(res1["labels"], res2["labels"])
