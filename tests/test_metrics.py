import pytest
import io
import pandas as pd
import numpy as np
from lab_modules.exp8_clustering.models import compute_silhouette, run_gmm
from core.report_generator import export_observations_csv

def test_silhouette_validity_bounds():
    X = np.random.rand(50, 2)
    score, msg = compute_silhouette(X, np.zeros(50, dtype=int))
    assert score is None
    assert "Requires >= 2 distinct clusters" in msg

def test_gmm_components_vs_clusters():
    X = np.random.rand(100, 2)
    res = run_gmm(X, n_components=5)
    assert res["configured_components"] == 5
    assert res["n_clusters"] <= 5

def test_clean_csv_export():
    obs = [{
        "run_id": "r1", "run_index": 1, "algorithm": "K-Means",
        "dataset_name": "Blobs (Separable)", "apply_scaling": True,
        "params": {"n_clusters": 3}, "metrics": {"silhouette": 0.67, "n_clusters": 3, "n_noise": 0}
    }]
    csv_str = export_observations_csv(obs)
    df = pd.read_csv(io.StringIO(csv_str))
    assert "Download as CSV" not in df.columns
    assert "Run_Number" in df.columns
    assert "Run_ID" in df.columns
    assert df.loc[0, "Run_Number"] == "Run 1"
