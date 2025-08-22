import requests
import gzip
import io
import pandas as pd
import os

def get_meteostat_hourly(station: str, year: int, save_path: str) -> pd.DataFrame:
    """
    Download hourly Meteostat data for a given station and year,
    save it to CSV, and return as DataFrame.

    Args:
        station (str): Meteostat station ID (e.g. '03772' for Paris-Orly).
        year (int): Year of interest (e.g. 2020).
        save_path (str): Path to save the CSV file.

    Returns:
        pd.DataFrame: Hourly weather data.
    """

    url = f"https://data.meteostat.net/hourly/{year}/{station}.csv.gz"
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. Status: {response.status_code}")

    # Decompress the GZ file
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
        df = pd.read_csv(gz)

    # Parse datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save CSV
    df.to_csv(save_path, index=False)
    print(f"Saved data to {save_path}")

    return df


# Example usage
if __name__ == "__main__":
    station_id = "07311"   # La Couarde
    year = 2023
    save_file = r"C:\Users\S1426582\Documents\GitHub\Solar_Home_Assistant_V2\data\consumption\raw\meteostat_{station_id}_{year}.csv"

    df = get_meteostat_hourly(station_id, year, save_file)
    print(df.head())
