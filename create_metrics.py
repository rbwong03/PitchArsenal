"""
This script creates metrics for pitch arsenals, including bridge scores, arsenal spread,
entropy, velocity buckets, pairwise velocity differential, and average fastball velocity.

Author: Ryan Wong
Date: April 25, 2025
"""
import pandas as pd
import numpy as np
from itertools import permutations, combinations
import os

def pairwise_velo_diff(group, min_pitches=300, min_usage=0.05):
    """
    Calculate the average pairwise velocity difference between pitch types for a given pitcher and season.
    Arguments:
        group (DataFrame): DataFrame containing pitch data for a specific pitcher and season.
        min_pitches (int): Minimum number of pitches required to calculate the metric.
        min_usage (float): Minimum usage percentage for a pitch type to be considered.
    Returns:
        float: Average pairwise velocity difference or NaN if conditions are not met.
    """
    pitch_velocities = group.set_index('pitch_type')['release_speed_mean'].to_dict()
    pitch_counts = group.set_index('pitch_type')['num_pitches'].to_dict()
    total_pitches = sum(pitch_counts.values())

    if total_pitches < min_pitches:
        return np.nan

    valid_pitch_types = [pt for pt, count in pitch_counts.items() if count / total_pitches >= min_usage]
    pitch_list = [pitch_velocities[pt] for pt in valid_pitch_types]

    if len(pitch_list) < 2:
        return np.nan

    pairwise_diffs = [
        abs(pitch_list[i] - pitch_list[j])
        for i in range(len(pitch_list))
        for j in range(i + 1, len(pitch_list))
    ]

    return np.mean(pairwise_diffs) if pairwise_diffs else np.nan

def calculate_entropy(group, usage_count_col='num_pitches', min_pitches=300):
    """
    Calculate the entropy of pitch usage for a given pitcher and season.
    Arguments:
        group (DataFrame): DataFrame containing pitch data for a specific pitcher and season.
        usage_count_col (str): Column name for pitch usage counts.
        min_pitches (int): Minimum number of pitches required to calculate the metric.
    Returns:
        float: Entropy value or NaN if conditions are not met."""
    pitch_counts = group.set_index('pitch_type')[usage_count_col].to_dict()
    total = sum(pitch_counts.values())
    if total < min_pitches:
        return np.nan

    probs = [count / total for count in pitch_counts.values() if count > 0]
    entropy = -sum(p * np.log2(p) for p in probs)
    return entropy

def assign_velocity_bucket(max_velo):
    """
    Assign a velocity bucket based on the maximum velocity.
    Arguments:
        max_velo (float): Maximum velocity of the pitch.
    Returns:
        str: Velocity bucket label.
    """
    if max_velo >= 97:
        return 'Elite Velo'
    elif max_velo >= 94:
        return 'High Velo'
    elif max_velo >= 90:
        return 'Average Velo'
    else:
        return 'Low Velo'

def calculate_average_fastball_velocity(df, fastball_types=['FF', 'SI', 'FT', 'FC']):
    """
    Calculate the average fastball velocity for each pitcher and season.
    Arguments:
        df (DataFrame): DataFrame containing pitch data.
        fastball_types (list): List of pitch types considered as fastballs.
    Returns:
        DataFrame: DataFrame containing average fastball velocities."""
    fastballs = df[df['pitch_type'].isin(fastball_types)]
    return fastballs.groupby(['pitcher', 'player_name', 'season'])['release_speed_mean'].mean().reset_index(name='avg_fastball_velocity')

def calculate_bridge_scores(df, group_cols, pfx_x_col='pfx_x_mean', pfx_z_col='pfx_z_mean',
                            usage_col='pitch_type', usage_count_col='num_pitches',
                            min_usage_rate=0.05,
                            line_scale=0.45, midpoint_scale=0.45, score_threshold=0.01):
    """
    Calculate the bridge scores for a given DataFrame.
    Arguments:
        df (DataFrame): DataFrame containing pitch data.
        group_cols (list): List of columns to group by.
        pfx_x_col (str): Column name for pfx_x values.
        pfx_z_col (str): Column name for pfx_z values.
        usage_col (str): Column name for pitch types.
        usage_count_col (str): Column name for pitch usage counts.
        min_usage_rate (float): Minimum usage percentage for a pitch type to be considered.
        line_scale (float): Scaling factor for line distance calculation.
        midpoint_scale (float): Scaling factor for midpoint distance calculation.
        score_threshold (float): Minimum score threshold to consider a triplet valid.
    Returns:
        DataFrame: DataFrame containing bridge scores and other metrics.
    """
    results = []

    for group_keys, group_data in df.groupby(group_cols):
        group_data = group_data.dropna(subset=[pfx_x_col, pfx_z_col, usage_col, usage_count_col])
        total_pitches = group_data[usage_count_col].sum()

        group_data['pitch_usage_pct'] = group_data[usage_count_col] / total_pitches
        group_data = group_data[group_data['pitch_usage_pct'] >= min_usage_rate]

        if group_data[usage_col].nunique() < 3:
            results.append(list(group_keys) + [0.0, 0.0])
            continue

        group_data = group_data.groupby(usage_col)[[pfx_x_col, pfx_z_col]].mean().reset_index()
        points = group_data[[pfx_x_col, pfx_z_col]].values
        valid_scores = []

        for i, j, k in permutations(range(len(points)), 3):
            A, C, B = points[i], points[j], points[k]
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

                line_score = np.exp(-dist_to_line / line_scale)
                midpoint_score = np.exp(-dist_to_midpoint / midpoint_scale)
                score = line_score * midpoint_score

                if score >= score_threshold:
                    valid_scores.append(score)

        total_score = np.sum(valid_scores)
        avg_score = np.mean(valid_scores) if valid_scores else 0.0
        results.append(list(group_keys) + [total_score, avg_score])

    columns = group_cols + ['total_bridge_score', 'average_bridge_quality']
    return pd.DataFrame(results, columns=columns)

def calculate_arsenal_spread(df, group_cols, 
                             pfx_x_col='pfx_x_mean', 
                             pfx_z_col='pfx_z_mean',
                             pitch_col='pitch_type', 
                             usage_count_col='num_pitches',
                             min_usage=0.05):
    """
    Calculate the spread of pitch types in a pitcher's arsenal.
    Arguments:
        df (DataFrame): DataFrame containing pitch data.
        group_cols (list): List of columns to group by.
        pfx_x_col (str): Column name for pfx_x values.
        pfx_z_col (str): Column name for pfx_z values.
        pitch_col (str): Column name for pitch types.
        usage_count_col (str): Column name for pitch usage counts.
        min_usage (float): Minimum usage percentage for a pitch type to be considered.
    Returns:
        DataFrame: DataFrame containing arsenal spread metrics.
    """
    spread_results = []

    for group_keys, group_data in df.groupby(group_cols):
        total_pitches = group_data[usage_count_col].sum()
        group_data['pitch_usage_pct'] = group_data[usage_count_col] / total_pitches
        filtered_data = group_data[group_data['pitch_usage_pct'] >= min_usage]

        grouped = filtered_data.groupby(pitch_col)[[pfx_x_col, pfx_z_col]].mean().dropna().reset_index()
        points = grouped[[pfx_x_col, pfx_z_col]].values
        unique_pitch_types = len(grouped)

        if unique_pitch_types < 2:
            spread_results.append(list(group_keys) + [np.nan, unique_pitch_types])
            continue

        pairwise_distances = [
            np.linalg.norm(points[i] - points[j])
            for i, j in combinations(range(len(points)), 2)
        ]
        avg_spread = np.mean(pairwise_distances)
        spread_results.append(list(group_keys) + [avg_spread, unique_pitch_types])

    columns = group_cols + ['arsenal_spread', 'num_pitch_types']
    return pd.DataFrame(spread_results, columns=columns)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate bridge and spread metrics for pitch arsenals.")
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned pitch summary CSV (e.g., qualified_pitch_summary.csv)")
    parser.add_argument("--bridge_output", type=str, default="data/bridge_scores.csv", help="Output path for bridge scores")
    parser.add_argument("--spread_output", type=str, default="data/arsenal_spread.csv", help="Output path for arsenal spread")
    parser.add_argument("--final_output", type=str, default="data/final_arsenal_metrics.csv", help="Output path for merged final metrics")

    args = parser.parse_args()
    df = pd.read_csv(args.input)

    bridge_scores_df = calculate_bridge_scores(
        df=df,
        group_cols=['pitcher', 'player_name', 'season'],
        pfx_x_col='pfx_x_mean',
        pfx_z_col='pfx_z_mean',
        usage_col='pitch_type',
        usage_count_col='num_pitches',
        min_usage_rate=0.05
    )
    os.makedirs(os.path.dirname(args.bridge_output), exist_ok=True)
    bridge_scores_df.to_csv(args.bridge_output, index=False)
    print(f"Bridge scores saved to {args.bridge_output}")

    arsenal_spread_df = calculate_arsenal_spread(
        df,
        group_cols=['pitcher', 'player_name', 'season'],
        pfx_x_col='pfx_x_mean',
        pfx_z_col='pfx_z_mean',
        pitch_col='pitch_type',
        usage_count_col='num_pitches',
        min_usage=0.05
    )
    arsenal_spread_df.to_csv(args.spread_output, index=False)
    print(f"Arsenal spread metrics saved to {args.spread_output}")

    entropy_df = df.groupby(['pitcher', 'player_name', 'season']).apply(
        calculate_entropy
    ).reset_index(name='pitch_entropy')

    df['pitch_usage_pct'] = df['num_pitches'] / df.groupby(['pitcher', 'player_name', 'season'])['num_pitches'].transform('sum')
    velo_df = df[df['pitch_usage_pct'] >= 0.05].groupby(['pitcher', 'player_name', 'season'])['release_speed_mean'].max().reset_index()
    velo_df['velo_bucket'] = velo_df['release_speed_mean'].apply(assign_velocity_bucket)

    avg_fb_velo_df = calculate_average_fastball_velocity(df)

    final_arsenal_metrics = pd.merge(
        bridge_scores_df,
        arsenal_spread_df,
        on=['pitcher', 'player_name', 'season'],
        how='inner'
    )

    final_arsenal_metrics = pd.merge(
        final_arsenal_metrics,
        entropy_df,
        on=['pitcher', 'player_name', 'season'],
        how='left'
    )

    final_arsenal_metrics = pd.merge(
        final_arsenal_metrics,
        velo_df[['pitcher', 'player_name', 'season', 'velo_bucket']],
        on=['pitcher', 'player_name', 'season'],
        how='left'
    )

    final_arsenal_metrics = pd.merge(
        final_arsenal_metrics,
        avg_fb_velo_df,
        on=['pitcher', 'player_name', 'season'],
        how='left'
    )

    velo_diff_df = df.groupby(['pitcher', 'player_name', 'season']).apply(
        pairwise_velo_diff
    ).reset_index(name='velocity_diff_unweighted')

    final_arsenal_metrics = pd.merge(
        final_arsenal_metrics,
        velo_diff_df,
        on=['pitcher', 'player_name', 'season'],
        how='left'
    )

    final_arsenal_metrics = pd.merge(
        final_arsenal_metrics,
        df[['pitcher', 'player_name', 'season', 'xwOBACON', 'csw_rate', 'xERA']].drop_duplicates(),
        on=['pitcher', 'player_name', 'season'],
        how='left'
    )

    final_arsenal_metrics = final_arsenal_metrics[[
        'pitcher', 'player_name', 'season', 'num_pitch_types', 'velo_bucket',
        'avg_fastball_velocity', 'total_bridge_score', 'average_bridge_quality',
        'arsenal_spread', 'pitch_entropy', 'velocity_diff_unweighted',
        'xwOBACON', 'csw_rate', 'xERA'
    ]]

    final_arsenal_metrics.sort_values('num_pitch_types', ascending=False, inplace=True)
    final_arsenal_metrics.to_csv(args.final_output, index=False)
    print(f"Final merged arsenal metrics saved to {args.final_output}")

    bucket_names = final_arsenal_metrics['velo_bucket'].dropna().unique()
    for bucket in bucket_names:
        df_bucket = final_arsenal_metrics[final_arsenal_metrics['velo_bucket'] == bucket]
        filename = f"data/final_arsenal_metrics_{bucket.replace(' ', '_')}.csv"
        df_bucket.to_csv(filename, index=False)
        print(f"Saved {bucket} metrics to {filename}")
