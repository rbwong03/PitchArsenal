"""
This creates a merged dataframe with all necessary data.

Author: Ryan Wong
Date: April 24, 2025
"""
import pandas as pd
import numpy as np
import os

def summarize_pitch_data(data):
    grouped = data.groupby(['pitcher', 'player_name', 'pitch_type', 'season']).agg(
        release_speed_mean=('release_speed', 'mean'),
        pfx_x_mean=('pfx_x', 'mean'),
        pfx_z_mean=('pfx_z', 'mean'),
        num_pitches=('release_speed', 'count')
    ).reset_index()

    pitcher_season_volume = grouped.groupby(['pitcher', 'player_name', 'season'])['num_pitches'].sum().reset_index()
    pitcher_season_volume = pitcher_season_volume[pitcher_season_volume['num_pitches'] >= 300]

    qualified = grouped.merge(
        pitcher_season_volume[['pitcher', 'player_name', 'season']],
        on=['pitcher', 'player_name', 'season'],
        how='inner'
    )

    return qualified

def convert_name_format(name):
    parts = name.strip().split()
    if len(parts) < 2:
        return name
    first = parts[0]
    last = ' '.join(parts[1:])
    return f"{last}, {first}"

def load_and_combine_seasons(seasons, input_dir="data"):
    dfs = []
    for year in seasons:
        path = os.path.join(input_dir, f"season_{year}.csv")
        print(f"Loading {path}")
        df = pd.read_csv(path)
        df['season'] = int(year)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def merge_external_metrics(df, seasons, data_dir="data"):
    csw_list, xera_list, xwoba_list = [], [], []

    for year in seasons:
        csw_path = os.path.join(data_dir, f"csw_by_pitcher_{year}.csv")
        if os.path.exists(csw_path):
            csw = pd.read_csv(csw_path)
            csw['season'] = int(year)
            csw_list.append(csw)

    xera_path = os.path.join(data_dir, "xera_by_pitcher.csv")
    if os.path.exists(xera_path):
        xera = pd.read_csv(xera_path)
        xera['player_name'] = xera['Name'].apply(convert_name_format)

    xwoba_path = os.path.join(data_dir, "xwobacon_by_pitcher.csv")
    if os.path.exists(xwoba_path):
        xwoba = pd.read_csv(xwoba_path)

    merged = df.copy()

    if csw_list:
        csw_all = pd.concat(csw_list, ignore_index=True)
        merged = merged.merge(csw_all, on=['pitcher', 'player_name', 'season'], how='left')

    if 'xera' in locals():
        merged = merged.merge(xera[['player_name', 'season', 'xERA']], on=['player_name', 'season'], how='left')

    if 'xwoba' in locals():
        merged = merged.merge(xwoba[['pitcher', 'player_name', 'season', 'xwOBACON']], on=['pitcher', 'player_name', 'season'], how='left')

    return merged

def main():
    data_dir = "data"
    seasons = ["2024"]
    output_path = f"{data_dir}/qualified_pitch_summary.csv"

    combined_data = load_and_combine_seasons(seasons, input_dir=data_dir)
    summarized = summarize_pitch_data(combined_data)
    merged = merge_external_metrics(summarized, seasons, data_dir=data_dir)

    merged.to_csv(output_path, index=False)
    print(f"Full pitch summary with CSW, xERA, xwOBACON saved to {output_path}")

if __name__ == "__main__":
    main()
