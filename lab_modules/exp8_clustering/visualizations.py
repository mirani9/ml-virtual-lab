import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_cluster_results(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray = None, title: str = "Cluster Visualization") -> go.Figure:
    df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    
    str_labels = [f"Cluster {l}" if l != -1 else "Noise Point (-1)" for l in labels]
    df["Cluster"] = str_labels
    
    symbols = ['diamond' if l == -1 else 'circle' for l in labels]
    df["Symbol"] = symbols
    
    fig = px.scatter(
        df, x="Feature 1", y="Feature 2", color="Cluster", symbol="Symbol",
        title=title, template="plotly_white", opacity=0.85,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        symbol_map={"diamond": "diamond", "circle": "circle"}
    )
    
    if centroids is not None:
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1], mode='markers',
            marker=dict(symbol='x', size=16, color='black', line=dict(width=3, color='yellow')),
            name='K-Means Centroids'
        ))
        
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_comparison_bar(results_dict: dict) -> go.Figure:
    algos, scores = [], []
    for name, res in results_dict.items():
        if res and res.get("silhouette") is not None:
            algos.append(name)
            scores.append(res["silhouette"])
            
    if not algos:
        fig = go.Figure()
        fig.add_annotation(text="No valid Silhouette Scores logged yet. Run algorithms on Workbench first.", showarrow=False, font=dict(size=14))
        fig.update_layout(template="plotly_white", height=250)
        return fig
        
    fig = px.bar(x=algos, y=scores, labels={'x': 'Algorithm', 'y': 'Silhouette Score'}, title="Silhouette Score Comparison", color=algos, template="plotly_white")
    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300, showlegend=False)
    return fig
