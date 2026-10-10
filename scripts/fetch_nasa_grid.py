"""
Fetch remaining NASA POWER grid points using requests library (more robust).
Skips the center point (already downloaded) and resumes from where it left off.
"""

import os
import sys
import json
import time
import csv
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MUPPANDAL_BBOX, NASA_DIR

BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS = "WS50M,WD50M,WS2M,T2M"
COMMUNITY = "RE"
START_DATE = "20230101"
END_DATE = "20231231"


def fetch_point(lat, lon, retries=3):
    """Fetch hourly data for a single point with retries."""
    url = (
        f"{BASE_URL}"
        f"?parameters={PARAMETERS}"
        f"&community={COMMUNITY}"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={START_DATE}"
        f"&end={END_DATE}"
        f"&format=JSON"
    )

    for attempt in range(retries):
        try:
            print(f"  Fetching lat={lat}, lon={lon} (attempt {attempt+1})...")
            resp = requests.get(url, timeout=180, headers={"User-Agent": "PawanSthal/1.0"})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    return None


def save_csv(data, path):
    """Convert NASA POWER JSON to CSV."""
    if not data or "properties" not in data:
        return 0

    params = data["properties"]["parameter"]
    param_names = list(params.keys())
    first_param = params[param_names[0]]
    timestamps = sorted(first_param.keys())

    rows = 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "year", "month", "day", "hour"] + param_names)
        for ts in timestamps:
            if len(ts) >= 10:
                row = [ts, ts[0:4], ts[4:6], ts[6:8], ts[8:10]]
                for p in param_names:
                    val = params[p].get(ts, -999)
                    row.append("" if val in (-999, -999.0) else val)
                writer.writerow(row)
                rows += 1
    return rows


def main():
    os.makedirs(NASA_DIR, exist_ok=True)
    bbox = MUPPANDAL_BBOX

    # Generate 3x3 grid
    points = []
    for i in range(3):
        for j in range(3):
            lat = round(bbox["south"] + (i + 0.5) * (bbox["north"] - bbox["south"]) / 3, 4)
            lon = round(bbox["west"] + (j + 0.5) * (bbox["east"] - bbox["west"]) / 3, 4)
            points.append((lat, lon))

    print("=" * 60)
    print("NASA POWER Grid Fetch (with resume)")
    print("=" * 60)

    grid_data = []
    for idx, (lat, lon) in enumerate(points):
        csv_path = os.path.join(NASA_DIR, f"grid_point_{idx:02d}_{lat}_{lon}.csv")

        # Skip if already downloaded
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
            print(f"  [{idx+1}/9] SKIP (already exists): {csv_path}")
            grid_data.append({"index": idx, "lat": lat, "lon": lon, "status": "ok"})
            continue

        print(f"\n  [{idx+1}/9] Fetching...")
        data = fetch_point(lat, lon)
        if data:
            rows = save_csv(data, csv_path)
            print(f"  Wrote {rows} rows -> {csv_path}")
            grid_data.append({"index": idx, "lat": lat, "lon": lon, "status": "ok"})
        else:
            print(f"  FAILED for point ({lat}, {lon})")
            grid_data.append({"index": idx, "lat": lat, "lon": lon, "status": "failed"})

        # Be polite
        if idx < len(points) - 1:
            time.sleep(3)

    # Save metadata
    meta = {
        "bbox": bbox,
        "period": f"{START_DATE}-{END_DATE}",
        "parameters": PARAMETERS.split(","),
        "points": grid_data
    }
    meta_path = os.path.join(NASA_DIR, "grid_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    ok = sum(1 for p in grid_data if p["status"] == "ok")
    print(f"\n{'='*60}")
    print(f"DONE: {ok}/{len(points)} grid points fetched.")
    print(f"Center point: already downloaded (8,760 rows)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
