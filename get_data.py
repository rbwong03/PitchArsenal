"""
This script downloads Statcast data for the specified seasons cleans the data, and adds xERA, xwOBACON, and CSW% stats.

Author: Ryan Wong
Date: April 24, 2025
"""
import pandas as pd
from pybaseball import statcast, pitching_stats
import os
import warnings

warnings.filterwarnings("ignore")

def download_season_data(start_date: str, end_date: str, year: str, output_dir="data"):
    """
    Download Statcast data for a given season and save it to a CSV file.

    Arguments:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        year (str): Year of the season.
        output_dir (str): Directory to save the CSV file.
    """
    print(f"Downloading Statcast data for {year}...")
    data = statcast(start_dt=start_date, end_dt=end_date)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"season_{year}.csv")
    data.to_csv(output_path, index=False)
    print(f"Saved {year} data to {output_path}")

def load_and_clean_season(year, input_dir="data"):
    """
    Load and clean the Statcast data for a given season.

    Arguments:
        year (str): Year of the season.
        input_dir (str): Directory where the CSV file is located.
    Returns:
        df: Cleaned DataFrame with relevant columns.
    """
    path = os.path.join(input_dir, f"season_{year}.csv")
    print(f"Loading {path}")
    df = pd.read_csv(path)

    important_cols = [
        'game_date', 'pitcher', 'player_name', 'pitch_type',
        'release_speed', 'pfx_x', 'pfx_z', 'description'
    ]
    df = df[important_cols].copy()
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    df = df.dropna(subset=['game_date'])  
    df['month'] = df['game_date'].dt.to_period('M')
    df['season'] = df['game_date'].dt.year

    return df

def compute_csw_rate(df, output_path="data/csw_by_pitcher.csv"):
    print("Computing CSW%...")
    csw_events = ['called_strike', 'swinging_strike', 'swinging_strike_blocked']
    df['is_csw'] = df['description'].isin(csw_events)

    csw_by_pitcher = (
        df.groupby(['pitcher', 'player_name', 'season'])['is_csw']
        .mean()
        .reset_index(name='csw_rate')
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    csw_by_pitcher.to_csv(output_path, index=False)
    print(f"CSW% data saved to {output_path}")

def fetch_xwobacon_data(seasons, output_path="data/xwobacon_by_pitcher.csv"):
    """
    Fetch xwOBACON data for the specified seasons and save to a CSV file.
    Arguments:
        seasons (list): List of seasons to fetch data for.
        output_path (str): Path to save the xwOBACON data.
    """
    xwobacon_dfs = []

    for year in seasons:
        print(f"Fetching xwOBACON for {year}...")
        try:
            data = statcast(f"{year}-03-28", f"{year}-10-05")
            data['season'] = year
            in_play = data[data['type'] == 'X'].dropna(subset=['estimated_woba_using_speedangle'])

            xwobacon = (
                in_play.groupby(['pitcher', 'player_name', 'season'])['estimated_woba_using_speedangle']
                .mean()
                .reset_index(name='xwOBACON')
            )
            xwobacon_dfs.append(xwobacon)

        except Exception as e:
            print(f"Error fetching xwOBACON for {year}: {e}")

    final_xwobacon = pd.concat(xwobacon_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_xwobacon.to_csv(output_path, index=False)
    print(f"xwOBACON data saved to {output_path}")

def fetch_xera_data(seasons, output_path="data/xera_by_pitcher.csv"):
    """
    Fetch xERA data for the specified seasons and save to a CSV file.
    Arguments:
        seasons (list): List of seasons to fetch data for.
        output_path (str): Path to save the xERA data.
    """
    xera_dfs = []

    for year in seasons:
        print(f"Fetching xERA for {year}...")
        try:
            stats = pitching_stats(year, qual=0)
            stats['season'] = year
            xera = stats[['Name', 'season', 'IP', 'xERA']].copy()
            xera_dfs.append(xera)
        except Exception as e:
            print(f"Error fetching xERA for {year}: {e}")

    final_xera = pd.concat(xera_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_xera.to_csv(output_path, index=False)
    print(f"xERA data saved to {output_path}")

def main():
    seasons = {
        "2024": ("2024-03-28", "2024-09-30")
    }

    all_cleaned_dfs = []
    for season_year, (start_date, end_date) in seasons.items():
        download_season_data(start_date, end_date, season_year)
        cleaned = load_and_clean_season(season_year)

        compute_csw_rate(cleaned, output_path=f"data/csw_by_pitcher_{season_year}.csv")

        all_cleaned_dfs.append(cleaned)

    combined_cleaned = pd.concat(all_cleaned_dfs, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    combined_cleaned.to_csv("data/combined_cleaned.csv", index=False)
    print("Combined cleaned data saved to data/combined_cleaned.csv")

    fetch_xwobacon_data(seasons=list(seasons.keys()))
    fetch_xera_data(seasons=list(seasons.keys()))

if __name__ == "__main__":
    main()
