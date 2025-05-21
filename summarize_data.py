"""
This script summarizes pitch data for each pitcher and season, calculating metrics such as
whiff rate and takes the averages of certain pitch data. It filters out pitchers with fewer than 300 pitches thrown in a season and saves the summarized data to a CSV file.
"""
import pandas as pd
import numpy as np

def calc_whiff_rate(descriptions):
    """
    Calculate the whiff rate based on pitch descriptions.
    Arguments:
        descriptions (Series): Series of pitch descriptions.
    Returns:
        float: Whiff rate as a percentage.
    """
    swings = descriptions.isin(['swinging_strike', 'swinging_strike_blocked'])
    swing_events = descriptions.isin([
        'swinging_strike', 'swinging_strike_blocked', 
        'foul', 'foul_tip', 'hit_into_play'
    ])
    if swing_events.sum() == 0:
        return np.nan
    return swings.sum() / swing_events.sum()


def summarize_pitch_data(data):
    """
    Summarize pitch data for each pitcher and season. This filters out pitchers
    with fewer than 300 pitches thrown in a season and calculates relevant metrics.
    Arguments:
        data (DataFrame): DataFrame containing pitch data.
    Returns:
        DataFrame: Summarized DataFrame with relevant metrics.
    """
    grouped = data.groupby(['pitcher', 'player_name', 'pitch_type', 'season']).agg(
        release_speed_mean=('release_speed', 'mean'),
        pfx_x_mean=('pfx_x', 'mean'),
        pfx_z_mean=('pfx_z', 'mean'),
        whiff_rate=('description', lambda x: calc_whiff_rate(x)),
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

def main():
    data = pd.read_csv("data/combined_cleaned.csv")
    summarized = summarize_pitch_data(data)
    summarized.to_csv("data/qualified_pitch_summary.csv", index=False)
    print("Pitch summary saved to data/qualified_pitch_summary.csv")

if __name__ == "__main__":
    main()
