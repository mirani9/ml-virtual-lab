AIM = """Perform clustering using K-Means, Expectation Maximization (EM/GMM), and DBSCAN algorithms and evaluate clustering quality using Silhouette Score and visualization techniques."""

OBJECTIVES = [
    "Understand the core differences between centroid-based (K-Means), distribution-based (EM/GMM), and density-based (DBSCAN) clustering.",
    "Evaluate the critical role of feature scaling in distance-sensitive clustering algorithms.",
    "Execute hyperparameter tuning for cluster counts (K), component distributions, epsilon (eps), and minimum samples.",
    "Quantify clustering performance using Silhouette Analysis and recognize when the metric is valid or misleading.",
    "Analyze dataset geometry to select the appropriate algorithm for non-spherical or noisy distributions."
]

THEORY = {
    "K-Means": "<b>Centroid-Based Clustering:</b> K-Means partitions N observations into K disjoint clusters by minimizing Within-Cluster Sum of Squares (Inertia).",
    "EM / GMM": "<b>Probabilistic Distribution-Based Clustering:</b> Gaussian Mixture Models assume data is generated from K Gaussian distributions using Expectation-Maximization.",
    "DBSCAN": "<b>Density-Based Spatial Clustering:</b> Groups points based on local density (eps, min_samples) and detects noise outliers without pre-specifying K."
}

PROCEDURE_STEPS = [
    {"step": 1, "title": "Select & Scale Dataset", "desc": "Choose a dataset geometry and decide whether to apply StandardScaler."},
    {"step": 2, "title": "Configure & Run K-Means", "desc": "Adjust cluster count K and initialization method."},
    {"step": 3, "title": "Configure & Run EM (GMM)", "desc": "Adjust components and covariance structure."},
    {"step": 4, "title": "Configure & Run DBSCAN", "desc": "Tune eps and min_samples."},
    {"step": 5, "title": "Compare & Record Observations", "desc": "Inspect side-by-side metric tables and log observations."}
]

PRE_LAB_QUIZ = [
    {
        "id": "q1",
        "question": "Why is feature scaling essential prior to running K-Means clustering?",
        "options": [
            "Select an option...",
            "Distance metrics are dominated by features with larger numerical scales",
            "Forces data into a perfectly Gaussian distribution",
            "Guarantees exact 10 iterations",
            "K-Means cannot run without scaling"
        ],
        "answer_idx": 1,
        "explanation": "Features with larger numerical scales dominate Euclidean distance calculations if left unscaled."
    },
    {
        "id": "q2",
        "question": "Which algorithm discovers arbitrary non-spherical shapes and identifies noise?",
        "options": [
            "Select an option...",
            "K-Means",
            "Standard GMM",
            "DBSCAN",
            "Hierarchical Agglomerative"
        ],
        "answer_idx": 3,
        "explanation": "DBSCAN forms clusters based on connected dense regions rather than spherical centroids."
    }
]

POST_LAB_QUIZ = [
    {
        "id": "pq1",
        "question": "Under what condition is the Silhouette Score invalid and uncomputable?",
        "options": [
            "Select an option...",
            "Score > 1.0",
            "When only 1 cluster is identified or all points are classified as noise",
            "When features are standardized",
            "When K = 3"
        ],
        "answer_idx": 2,
        "explanation": "Evaluating inter vs intra-cluster distance requires at least 2 distinct non-noise clusters."
    },
    {
        "id": "pq2",
        "question": "How does EM/GMM differ fundamentally from K-Means?",
        "options": [
            "Select an option...",
            "Assigns soft membership probabilities rather than hard clusters",
            "Requires eps parameter",
            "Runs faster on massive datasets",
            "Cannot handle 2D data"
        ],
        "answer_idx": 1,
        "explanation": "GMM computes probabilistic soft assignments for each cluster component."
    }
]
