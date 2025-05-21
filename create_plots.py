"""
This script creates plots of pitch movement for a specified pitcher and season,
using the bridge metric to identify valid triplets of pitches. It filters the data
based on usage and computes the average movement for each pitch type.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
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
