import time
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def compute_silhouette(X: np.ndarray, labels: np.ndarray, exclude_noise: bool = True) -> tuple[float | None, str]:
    if X is None or len(X) == 0 or len(labels) != len(X):
        return None, "Not applicable: Empty or mismatched feature/label vectors."

    if exclude_noise:
        valid_mask = labels != -1
        eval_X = X[valid_mask]
        eval_labels = labels[valid_mask]
    else:
        eval_X = X
        eval_labels = labels

    if len(eval_X) == 0:
        return None, "Not applicable: All samples classified as noise (-1)."

    unique_labels = np.unique(eval_labels)
    n_unique = len(unique_labels)
    n_samples = len(eval_X)

    if n_unique < 2:
        return None, f"Not applicable: Requires >= 2 distinct clusters (found {n_unique})."
    
    if n_unique >= n_samples:
        return None, f"Not applicable: Cluster count ({n_unique}) matches sample count."

    try:
        score = float(silhouette_score(eval_X, eval_labels))
        return round(score, 4), "Valid metric."
    except Exception as e:
        return None, f"Calculation error: {str(e)}"

def run_kmeans(X: np.ndarray, n_clusters: int = 3, init: str = "k-means++", max_iter: int = 300, random_state: int = 42) -> dict:
    start_time = time.perf_counter()
    model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    exec_time = (time.perf_counter() - start_time) * 1000
    
    sil_score, sil_msg = compute_silhouette(X, labels)
    
    return {
        "algorithm": "K-Means",
        "labels": labels,
        "centroids": model.cluster_centers_,
        "inertia": float(model.inertia_),
        "n_clusters": len(np.unique(labels)),
        "configured_components": n_clusters,
        "n_noise": 0,
        "noise_ratio": 0.0,
        "silhouette": sil_score,
        "silhouette_msg": sil_msg,
        "exec_time_ms": round(exec_time, 2),
        "params": {"n_clusters": n_clusters, "init": init, "max_iter": max_iter, "random_state": random_state}
    }

def run_gmm(X: np.ndarray, n_components: int = 3, covariance_type: str = "full", random_state: int = 42) -> dict:
    start_time = time.perf_counter()
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    labels = model.fit_predict(X)
    exec_time = (time.perf_counter() - start_time) * 1000
    
    assigned_clusters = len(np.unique(labels))
    sil_score, sil_msg = compute_silhouette(X, labels)
    
    return {
        "algorithm": "EM (GMM)",
        "labels": labels,
        "probabilities": model.predict_proba(X),
        "bic": float(model.bic(X)),
        "aic": float(model.aic(X)),
        "n_clusters": assigned_clusters,
        "configured_components": n_components,
        "n_noise": 0,
        "noise_ratio": 0.0,
        "silhouette": sil_score,
        "silhouette_msg": sil_msg,
        "exec_time_ms": round(exec_time, 2),
        "params": {"n_components": n_components, "covariance_type": covariance_type, "random_state": random_state}
    }

def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5, exclude_noise: bool = True) -> dict:
    start_time = time.perf_counter()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    exec_time = (time.perf_counter() - start_time) * 1000
    
    noise_count = int(np.sum(labels == -1))
    non_noise_clusters = len(np.unique(labels[labels != -1]))
    sil_score, sil_msg = compute_silhouette(X, labels, exclude_noise=exclude_noise)
    
    return {
        "algorithm": "DBSCAN",
        "labels": labels,
        "n_clusters": non_noise_clusters,
        "configured_components": "N/A",
        "n_noise": noise_count,
        "noise_ratio": round((noise_count / len(X)) * 100, 2),
        "silhouette": sil_score,
        "silhouette_msg": sil_msg,
        "exec_time_ms": round(exec_time, 2),
        "params": {"eps": eps, "min_samples": min_samples}
    }
