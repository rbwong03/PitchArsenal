"""
This script creates plots of pitch movement for a specified pitcher and season,
using the bridge metric to identify valid triplets of pitches. It also plots kmeans clustering and plotting distributions of the clusters of certain data columns. It filters the data
based on usage and computes the average movement for each pitch type. 

Author: Ryan Wong
Date: April 27, 2025
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os


def plot_bridge_pitch_clusters(
    raw_data,
    pitcher_name,
    season,
    alpha=0.1,
    max_midpoint_dist=0.3,
    min_usage=0.05,
    metric_name="bridge_score",
    save=True
):
    """
    Plot the pitch movement for a specified pitcher and season, using the bridge metric
    to identify valid triplets of pitches. The plot shows the average movement for each
    pitch type and highlights the triplet connections.
    Arguments:
        raw_data (DataFrame): DataFrame containing pitch data.
        pitcher_name (str): Name of the pitcher to plot.
        season (int): Season year to filter the data.
        alpha (float): Threshold for triplet connection.
        max_midpoint_dist (float): Maximum distance from midpoint for triplet connection.
        min_usage (float): Minimum usage percentage for pitch types.
        metric_name (str): Name of the metric used for saving the plot.
        save (bool): Whether to save the plot as a PDF file.
    """
    pitcher_data = raw_data[
        (raw_data['player_name'] == pitcher_name) & 
        (raw_data['season'] == season)
    ]

    if pitcher_data.empty:
        print(f"No data for {pitcher_name} in {season}")
        return

    # Filter by usage threshold
    pitch_counts = pitcher_data['pitch_type'].value_counts(normalize=True)
    valid_pitch_types = pitch_counts[pitch_counts >= min_usage].index
    pitcher_data = pitcher_data[pitcher_data['pitch_type'].isin(valid_pitch_types)]

    if pitcher_data['pitch_type'].nunique() < 3:
        print(f"Not enough pitch types to form triplets for {pitcher_name} in {season}")
        return

    # Compute pitch type means
    means = pitcher_data.groupby('pitch_type').agg({
        'pfx_x': 'mean',
        'pfx_z': 'mean'
    }).reset_index()

    coords = means[['pfx_x', 'pfx_z']].values
    labels = means['pitch_type'].values

    # Plot base scatter
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=pitcher_data,
        x='pfx_x', y='pfx_z',
        hue='pitch_type',
        alpha=0.6,
        palette='tab10',
        s=20
    )

    # Plot pitch type centers
    plt.scatter(
        means['pfx_x'], means['pfx_z'],
        color='black', marker='*', s=300, label='Pitch Type Center'
    )

    # Evaluate triplets
    for i, j, k in permutations(range(len(coords)), 3):
        if k in (i, j):
            continue

        A = coords[i]
        C = coords[j]
        B = coords[k]

        AC = C - A
        AB = B - A
        norm_AC_squared = np.dot(AC, AC)
        if norm_AC_squared == 0:
            continue

        proj = np.dot(AB, AC) / norm_AC_squared
        if 0 <= proj <= 1:
            closest_point = A + proj * AC
            dist_to_line = np.linalg.norm(B - closest_point)
            midpoint = (A + C) / 2
            dist_to_midpoint = np.linalg.norm(B - midpoint)

            if (dist_to_line / np.linalg.norm(AC) <= alpha and dist_to_midpoint <= max_midpoint_dist):
                # A–C (outer)
                plt.plot([A[0], C[0]], [A[1], C[1]], 'k--', alpha=0.3)
                # A–B and C–B
                plt.plot([A[0], B[0]], [A[1], B[1]], 'k--', alpha=0.5)
                plt.plot([C[0], B[0]], [C[1], B[1]], 'k--', alpha=0.5)
                # Bridge pitch to midpoint
                plt.plot([B[0], midpoint[0]], [B[1], midpoint[1]], 'r--', alpha=0.7)

    plt.title(f'Pitch Arsenal for {pitcher_name} ({season})')
    plt.xlabel('Horizontal Movement (pfx_x)')
    plt.ylabel('Vertical Movement (pfx_z)')
    plt.axhline(0, color='gray', linestyle='--', lw=1)
    plt.axvline(0, color='gray', linestyle='--', lw=1)
    plt.grid(True)
    plt.legend(title='Pitch Type', loc='best')
    plt.tight_layout()

    if save:
        folder = f"{metric_name}_plots"
        os.makedirs(folder, exist_ok=True)
        safe_name = pitcher_name.replace(", ", "_").replace(" ", "_")
        filepath = os.path.join(folder, f"{safe_name}_{season}.pdf")
        plt.savefig(filepath)
        print(f"Saved plot to {filepath}")

    plt.close()


def plot_elbow_and_silhouette(df, features, save_path='plots/clustering', k_range=range(2, 15)):
    """
    Plot elbow method (inertia) and silhouette score side-by-side for a range of KMeans clusters.

    Parameters:
        df (pd.DataFrame): DataFrame containing features for clustering.
        features (list): List of column names to use for clustering.
        save_path (str): Directory to save the output plot.
        k_range (range): Range of cluster values to try (e.g., range(2, 15)).
    """
    os.makedirs(save_path, exist_ok=True)

    X = df[features].dropna().copy()

    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Clustering Inertia', color=color)
    ax1.plot(k_range, inertias, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('KMeans Clustering: Elbow and Silhouette Analysis')

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette_scores, marker='s', linestyle='--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.grid(True)
    plot_path = os.path.join(save_path, 'elbow_silhouette_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")



def run_kmeans_clustering_and_plotting(input_csv="data/final_arsenal_metrics.csv", k=5):
    """
    Run KMeans clustering on pitcher arsenal metrics and plot the results.
    Parameters:
        input_csv (str): Path to the input CSV file containing pitcher metrics.
        k (int): Number of clusters for KMeans.
    Returns:
        df (pd.DataFrame): DataFrame with cluster assignments and PCA components.
    """
    df = pd.read_csv(input_csv)
    df[['total_bridge_score', 'average_bridge_quality']] = df[['total_bridge_score', 'average_bridge_quality']].fillna(0)

    features = [
        'total_bridge_score',
        'arsenal_spread',
        'pitch_entropy',
        'velocity_diff_unweighted',
    ]

    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df.loc[X.index, 'cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df.loc[X.index, 'pca1'] = components[:, 0]
    df.loc[X.index, 'pca2'] = components[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.loc[X.index], x='pca1', y='pca2', hue='cluster', palette='tab10', s=80)
    plt.title("KMeans Clusters of Pitcher Arsenal Profiles (PCA-reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("plots/clustering", exist_ok=True)
    plt.savefig("plots/clustering/kmeans_clusters.pdf")
    df.to_csv("data/clustered_pitchers.csv", index=False)
    print("Saved cluster plot to plots/clustering/kmeans_clusters.pdf")

    plt.show()

    return df


def plot_feature_distributions_by_cluster(df, features, cluster_col='cluster', save_path='plots/clustering'):
    """
    Plot feature distributions by cluster using violin plots.
    Parameters:
        df (pd.DataFrame): DataFrame containing features and cluster assignments.
        features (list): List of feature names to plot.
        cluster_col (str): Column name for cluster assignments.
        save_path (str): Directory to save the output plots.
    """
    os.makedirs(save_path, exist_ok=True)

    df[cluster_col] = pd.to_numeric(df[cluster_col], errors='coerce')
    df = df[df[cluster_col].notna()].copy()
    df[cluster_col] = df[cluster_col].astype(int)
    sorted_clusters = sorted(df[cluster_col].unique())

    for feature in features:
        plot_df = df[[cluster_col, feature]].dropna().copy()

        plot_df[cluster_col] = pd.Categorical(plot_df[cluster_col], categories=sorted_clusters, ordered=True)

        plt.figure(figsize=(8, 6))
        sns.violinplot(
            x=cluster_col,
            y=feature,
            data=plot_df,
            order=sorted_clusters,
            inner='box'
        )
        plt.title(f'{feature} by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(feature)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'{save_path}/violin_{feature}_by_cluster.png')
        plt.close()
