AIM = "Perform clustering analysis using K-Means, Expectation Maximization (EM/GMM), and DBSCAN algorithms on diverse data topologies, and evaluate their partitioning performance using the Silhouette Score metric and visual diagnostics."

OBJECTIVES = [
    "Understand the underlying mathematical mechanics and geometric assumptions of K-Means, Expectation-Maximization (GMM), and DBSCAN.",
    "Analyze the impact of feature scaling (StandardScaler) on distance-based clustering metrics.",
    "Evaluate cluster quality using Silhouette Coefficient analysis across convex and non-convex geometric topologies.",
    "Identify noise/outlier points in density-based spatial partitioning.",
    "Generate comprehensive, publication-ready experiment reports detailing clustering metrics and performance summaries."
]

THEORY = {
    "K-Means": """
### 1. K-Means Clustering Algorithm

**Overview:**
K-Means is an iterative, distance-based partitioning algorithm that divides $N$ observations into $K$ distinct, non-overlapping clusters. Each cluster is defined by its centroid (the mean of points belonging to that cluster).

#### Mathematical Formulation & Objective Function:
The algorithm seeks to minimize the **Within-Cluster Sum of Squares (WCSS)** or Inertia:

$$J = \\sum_{k=1}^{K} \\sum_{x_i \\in C_k} ||x_i - \\mu_k||^2$$

Where:
- $C_k$ is the set of points assigned to cluster $k$.
- $\\mu_k = \\frac{1}{|C_k|} \\sum_{x_i \\in C_k} x_i$ is the centroid of cluster $C_k$.
- $||x_i - \\mu_k||$ represents the Euclidean distance.

#### Algorithmic Steps:
1. **Initialization:** Select $K$ initial cluster centroids (using random selection or $K\\text{-means++}$).
2. **Assignment Step:** Assign each data point $x_i$ to the nearest centroid:
   $$c^{(i)} := \\arg\\min_k ||x_i - \\mu_k||^2$$
3. **Update Step:** Recalculate the centroid $\\mu_k$ of each cluster based on assigned points.
4. **Convergence Check:** Repeat steps 2 and 3 until centroids stop changing or `max_iter` is reached.

#### Strengths & Limitations:
- **Pros:** Computationally efficient ($O(N \\cdot K \\cdot d)$ per iteration), scalable to large datasets.
- **Cons:** Assumes spherical, convex cluster geometries of similar size; highly sensitive to initial centroid placement and outliers.
""",

    "EM / GMM": """
### 2. Expectation-Maximization (EM) with Gaussian Mixture Models (GMM)

**Overview:**
Gaussian Mixture Models (GMM) treat clustering as a probabilistic density estimation problem. Unlike K-Means (which performs hard assignment), GMM performs **soft clustering**, assigning each point a probability of belonging to each Gaussian distribution component.

#### Mathematical Foundation:
A GMM models the data distribution as a linear combination of $K$ Gaussian distributions:

$$p(x) = \\sum_{k=1}^{K} \\pi_k \\mathcal{N}(x | \\mu_k, \\Sigma_k)$$

Where:
- $\\pi_k$ is the mixture weight for component $k$ (with $\\sum \\pi_k = 1$).
- $\\mathcal{N}(x | \\mu_k, \\Sigma_k)$ is the multivariate Gaussian distribution with mean vector $\\mu_k$ and covariance matrix $\\Sigma_k$.

#### Expectation-Maximization (EM) Algorithm:
1. **E-Step (Expectation):** Calculate responsibilities $\\gamma_{ik}$ (the posterior probability that point $x_i$ belongs to component $k$):
   $$\\gamma_{ik} = \\frac{\\pi_k \\mathcal{N}(x_i | \\mu_k, \\Sigma_k)}{\\sum_{j=1}^{K} \\pi_j \\mathcal{N}(x_i | \\mu_j, \\Sigma_j)}$$

2. **M-Step (Maximization):** Update Gaussian parameters using the computed responsibilities:
   $$N_k = \\sum_{i=1}^{N} \\gamma_{ik}$$
   $$\\mu_k^{new} = \\frac{1}{N_k} \\sum_{i=1}^{N} \\gamma_{ik} x_i$$
   $$\\Sigma_k^{new} = \\frac{1}{N_k} \\sum_{i=1}^{N} \\gamma_{ik} (x_i - \\mu_k^{new})(x_i - \\mu_k^{new})^T$$
   $$\\pi_k^{new} = \\frac{N_k}{N}$$

#### Strengths & Limitations:
- **Pros:** Flexible covariance options (`full`, `tied`, `diag`, `spherical`) accommodate elliptical cluster shapes; provides probabilistic cluster memberships.
- **Cons:** Can converge to local optima; computationally intensive for high-dimensional feature spaces.
""",

    "DBSCAN": """
### 3. Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

**Overview:**
DBSCAN groups together points that are closely packed (high-density areas) and marks points in low-density regions as noise/outliers. Unlike K-Means and GMM, DBSCAN does not require specifying the number of clusters $K$ in advance and can discover clusters of arbitrary, non-spherical shapes.

#### Key Hyperparameters:
- **$\\epsilon$ (eps):** The maximum radius distance defining a point's neighborhood.
- **`min_samples`:** The minimum number of points required within the $\\epsilon$-neighborhood to form a dense region.

#### Point Classification Rules:
- **Core Point:** A point $p$ with at least `min_samples` points within its $\\epsilon$-neighborhood ($|N_\\epsilon(p)| \\ge \\text{min\\_samples}$).
- **Border Point:** A point reachable from a core point within distance $\\epsilon$, but having fewer than `min_samples` points in its own neighborhood.
- **Noise Point (-1):** Any point that is neither a core point nor a border point.

#### Quality Evaluation: Silhouette Score
The Silhouette Score $s(i)$ measures how similar a point is to its own cluster compared to other clusters:

$$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$$

Where:
- $a(i)$ is the mean intra-cluster distance of point $i$.
- $b(i)$ is the mean nearest-cluster distance for point $i$.
- $s(i) \\in [-1, 1]$: Scores near $+1$ indicate well-separated clusters; scores near $0$ indicate overlapping boundaries; negative values indicate misclassifications.
"""
}

PROCEDURE_STEPS = [
    {"step": 1, "title": "Review Theory & Concepts", "desc": "Read the foundational mechanics of K-Means, GMM, and DBSCAN."},
    {"step": 2, "title": "Complete Readiness Quiz", "desc": "Test your baseline knowledge before entering the experimental workbench."},
    {"step": 3, "title": "Select Dataset & Preprocessing", "desc": "Choose data geometry topology, sample size, and feature scaling parameters."},
    {"step": 4, "title": "Execute Clustering Algorithms", "desc": "Tune parameters for K-Means, GMM, and DBSCAN and run partitioning."},
    {"step": 5, "title": "Analyze Silhouette Metrics", "desc": "Compare algorithm performance via scatter plots and bar charts in the observation log."},
    {"step": 6, "title": "Generate Final Practical Report", "desc": "Complete the post-lab assessment and export your official PDF practical report."}
]

PRE_LAB_QUIZ = [
    {
        "id": "q1",
        "question": "Which clustering algorithm is capable of discovering arbitrary non-spherical shapes and identifying noise points?",
        "options": ["Select an option...", "K-Means", "DBSCAN", "Gaussian Mixture Model (GMM)", "Hierarchical Agglomerative"],
        "answer_idx": 2,
        "explanation": "DBSCAN relies on density connectivity and can trace non-convex shapes like moons or circles while labeling sparse points as noise (-1)."
    },
    {
        "id": "q2",
        "question": "What does a Silhouette Score of +0.85 indicate for a clustering assignment?",
        "options": ["Select an option...", "Clusters are dense and well-separated", "Points are misclassified into wrong clusters", "Clusters overlap completely", "All samples are marked as noise"],
        "answer_idx": 1,
        "explanation": "A Silhouette score near +1.0 indicates strong intra-cluster compactness and clear separation from neighboring clusters."
    }
]

POST_LAB_QUIZ = [
    {
        "id": "pq1",
        "question": "Why does standard K-Means often fail on Concentric Circles datasets?",
        "options": ["Select an option...", "K-Means assumes convex/spherical cluster geometries using Euclidean centroids", "K-Means requires labeled training data", "K-Means cannot calculate distance metrics", "K-Means requires density threshold parameters"],
        "answer_idx": 1,
        "explanation": "K-Means minimizes within-cluster variance using linear Voronoi boundaries, which cannot capture nested circular structures."
    }
]
