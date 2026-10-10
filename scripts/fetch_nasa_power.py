"""
Fetch hourly wind data from NASA POWER API for Muppandal cluster.
Saves CSV files to data/nasa_power/

NASA POWER API: https://power.larc.nasa.gov/
- No API key required
- Free for research and commercial use
- Parameters: WS50M (wind speed 50m), WD50M (wind direction 50m), T2M (temp 2m)
"""

import os
import sys
import json
import time
import csv
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MUPPANDAL_BBOX, NASA_DIR

# NASA POWER API configuration
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS = "WS50M,WD50M,WS2M,T2M"
COMMUNITY = "RE"

# We'll fetch 1 full year of data (2023) for the wind rose
# NASA POWER hourly data is available with ~2 month lag
START_DATE = "20230101"
END_DATE = "20231231"


def fetch_nasa_power_point(lat, lon, start_date, end_date, parameters):
    """Fetch hourly data for a single point from NASA POWER API."""
    url = (
        f"{BASE_URL}"
        f"?parameters={parameters}"
        f"&community={COMMUNITY}"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start_date}"
        f"&end={end_date}"
        f"&format=JSON"
    )

    print(f"  Fetching: lat={lat}, lon={lon}, {start_date}–{end_date}")
    print(f"  URL: {url}")

    req = Request(url, headers={"User-Agent": "PawanSthal/1.0"})

    try:
        with urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode())
        return data
    except HTTPError as e:
        print(f"  HTTP Error {e.code}: {e.reason}")
        return None
    except URLError as e:
        print(f"  URL Error: {e.reason}")
        return None


def nasa_json_to_csv(data, output_path):
    """Convert NASA POWER JSON response to a flat CSV."""
    if not data or "properties" not in data:
        print("  ERROR: No valid data in response")
        return False

    params = data["properties"]["parameter"]
    param_names = list(params.keys())

    # NASA POWER hourly format: keys are "YYYYMMDD" or "YYYYMMDDHH"
    # Get all timestamps from the first parameter
    first_param = params[param_names[0]]
    timestamps = sorted(first_param.keys())

    rows_written = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "year", "month", "day", "hour"] + param_names)

        for ts in timestamps:
            # Parse timestamp (format: YYYYMMDDHH)
            if len(ts) >= 10:
                year = ts[0:4]
                month = ts[4:6]
                day = ts[6:8]
                hour = ts[8:10] if len(ts) >= 10 else "00"
            else:
                continue

            row = [ts, year, month, day, hour]
            valid = True
            for param in param_names:
                val = params[param].get(ts, -999)
                # NASA uses -999 for fill values
                if val == -999 or val == -999.0:
                    val = ""
                row.append(val)

            writer.writerow(row)
            rows_written += 1

    print(f"  Wrote {rows_written} rows to {output_path}")
    return True


def fetch_grid_points():
    """
    Fetch data for a grid of points across the Muppandal bbox.
    This gives us spatial coverage for the wind map.
    """
    bbox = MUPPANDAL_BBOX
    # Create a 3×3 grid of points (9 points total — API-friendly)
    lat_steps = 3
    lon_steps = 3
    lat_range = bbox["north"] - bbox["south"]
    lon_range = bbox["east"] - bbox["west"]

    points = []
    for i in range(lat_steps):
        for j in range(lon_steps):
            lat = bbox["south"] + (i + 0.5) * lat_range / lat_steps
            lon = bbox["west"] + (j + 0.5) * lon_range / lon_steps
            points.append((round(lat, 4), round(lon, 4)))

    return points


def main():
    os.makedirs(NASA_DIR, exist_ok=True)

    print("=" * 60)
    print("PawanSthal — NASA POWER Data Fetcher")
    print("=" * 60)
    print(f"Cluster: {MUPPANDAL_BBOX['name']}")
    print(f"Bbox: S={MUPPANDAL_BBOX['south']}, N={MUPPANDAL_BBOX['north']}, "
          f"W={MUPPANDAL_BBOX['west']}, E={MUPPANDAL_BBOX['east']}")
    print(f"Period: {START_DATE} – {END_DATE}")
    print(f"Parameters: {PARAMETERS}")
    print()

    # 1. Fetch center point (primary data for wind rose)
    print("[1/2] Fetching CENTER point data (for wind rose)...")
    center_data = fetch_nasa_power_point(
        MUPPANDAL_BBOX["center_lat"],
        MUPPANDAL_BBOX["center_lon"],
        START_DATE, END_DATE, PARAMETERS
    )

    if center_data:
        center_csv = os.path.join(NASA_DIR, "muppandal_center_hourly.csv")
        nasa_json_to_csv(center_data, center_csv)

        # Also save raw JSON for reference
        raw_json = os.path.join(NASA_DIR, "muppandal_center_raw.json")
        with open(raw_json, "w") as f:
            json.dump(center_data, f, indent=2)
        print(f"  Raw JSON saved to {raw_json}")
    else:
        print("  FAILED to fetch center point data!")

    print()

    # 2. Fetch grid points (for spatial wind map)
    print("[2/2] Fetching GRID points (3×3 grid for spatial coverage)...")
    grid_points = fetch_grid_points()
    print(f"  Grid points: {len(grid_points)}")

    all_grid_data = []
    for idx, (lat, lon) in enumerate(grid_points):
        print(f"\n  Point {idx + 1}/{len(grid_points)}:")
        point_data = fetch_nasa_power_point(lat, lon, START_DATE, END_DATE, PARAMETERS)

        if point_data:
            csv_path = os.path.join(NASA_DIR, f"grid_point_{idx:02d}_{lat}_{lon}.csv")
            nasa_json_to_csv(point_data, csv_path)
            all_grid_data.append({
                "index": idx,
                "lat": lat,
                "lon": lon,
                "file": csv_path,
                "status": "ok"
            })
        else:
            all_grid_data.append({
                "index": idx,
                "lat": lat,
                "lon": lon,
                "status": "failed"
            })

        # Be polite to the API
        if idx < len(grid_points) - 1:
            print("  Waiting 2s before next request...")
            time.sleep(2)

    # Save grid metadata
    meta_path = os.path.join(NASA_DIR, "grid_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "bbox": MUPPANDAL_BBOX,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "parameters": PARAMETERS.split(","),
            "points": all_grid_data
        }, f, indent=2)
    print(f"\n  Grid metadata saved to {meta_path}")

    # Summary
    ok_count = sum(1 for p in all_grid_data if p["status"] == "ok")
    print(f"\n{'=' * 60}")
    print(f"DONE. Center point + {ok_count}/{len(grid_points)} grid points fetched.")
    print(f"Data saved to: {NASA_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
