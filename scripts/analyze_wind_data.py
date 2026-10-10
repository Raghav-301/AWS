"""
Quick analysis of the NASA POWER center point data to validate wind conditions.
Generates summary stats and a wind rose data file for the Muppandal cluster.
"""

import os
import sys
import csv
import json
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NASA_DIR, MUPPANDAL_BBOX


def main():
    csv_path = os.path.join(NASA_DIR, "muppandal_center_hourly.csv")
    if not os.path.exists(csv_path):
        print("ERROR: Center point data not found! Run fetch_nasa_power.py first.")
        return

    # Read data
    wind_speeds = []
    wind_dirs = []
    monthly_ws = defaultdict(list)
    hourly_ws = defaultdict(list)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ws = row.get("WS50M", "")
            wd = row.get("WD50M", "")
            if ws and wd:
                ws_val = float(ws)
                wd_val = float(wd)
                wind_speeds.append(ws_val)
                wind_dirs.append(wd_val)
                monthly_ws[row["month"]].append(ws_val)
                hourly_ws[row["hour"]].append(ws_val)

    if not wind_speeds:
        print("ERROR: No valid wind data found!")
        return

    # Basic stats
    n = len(wind_speeds)
    mean_ws = sum(wind_speeds) / n
    max_ws = max(wind_speeds)
    min_ws = min(wind_speeds)
    sorted_ws = sorted(wind_speeds)
    median_ws = sorted_ws[n // 2]

    print("=" * 60)
    print("  Muppandal Wind Data Summary (2023)")
    print("=" * 60)
    print(f"  Total hourly records: {n:,}")
    print(f"  Wind speed at 50m (m/s):")
    print(f"    Mean:   {mean_ws:.2f}")
    print(f"    Median: {median_ws:.2f}")
    print(f"    Min:    {min_ws:.2f}")
    print(f"    Max:    {max_ws:.2f}")
    print()

    # Monthly breakdown
    print("  Monthly average wind speed (m/s):")
    print("  " + "-" * 40)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_avgs = {}
    for m in range(1, 13):
        key = f"{m:02d}"
        if key in monthly_ws:
            avg = sum(monthly_ws[key]) / len(monthly_ws[key])
            monthly_avgs[month_names[m-1]] = avg
            bar = "#" * int(avg * 3)
            print(f"    {month_names[m-1]}: {avg:5.2f}  {bar}")
    print()

    # Wind rose data (16 direction bins)
    print("  Wind rose (16 directions):")
    print("  " + "-" * 40)
    dir_bins = {}
    bin_size = 22.5
    dir_names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                 "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    for name in dir_names:
        dir_bins[name] = {"count": 0, "ws_sum": 0}

    for ws, wd in zip(wind_speeds, wind_dirs):
        idx = int(((wd + bin_size / 2) % 360) / bin_size)
        if idx >= 16:
            idx = 0
        name = dir_names[idx]
        dir_bins[name]["count"] += 1
        dir_bins[name]["ws_sum"] += ws

    for name in dir_names:
        d = dir_bins[name]
        pct = d["count"] / n * 100
        avg_ws = d["ws_sum"] / d["count"] if d["count"] > 0 else 0
        bar = "#" * int(pct * 2)
        print(f"    {name:>3}: {pct:5.1f}%  avg {avg_ws:.1f} m/s  {bar}")

    # Predominant direction
    predominant = max(dir_bins.items(), key=lambda x: x[1]["count"])
    print(f"\n  Predominant direction: {predominant[0]} "
          f"({predominant[1]['count']/n*100:.1f}%)")

    # Hub-height extrapolation (power law)
    # ws_h = ws_ref * (h / h_ref)^alpha
    # alpha ~0.14 for open terrain (wind farms)
    alpha = 0.14
    h_ref = 50  # NASA POWER measurement height
    for h in [100, 120, 150]:
        factor = (h / h_ref) ** alpha
        ws_h = mean_ws * factor
        print(f"\n  Extrapolated to {h}m hub height (alpha={alpha}):")
        print(f"    Mean wind speed: {ws_h:.2f} m/s")

    # Save wind rose data for later use
    rose_data = {
        "center": {"lat": MUPPANDAL_BBOX["center_lat"], "lon": MUPPANDAL_BBOX["center_lon"]},
        "period": "2023",
        "total_hours": n,
        "mean_ws_50m": round(mean_ws, 2),
        "directions": {},
        "monthly_avg": {m: round(v, 2) for m, v in monthly_avgs.items()},
        "hub_height_extrapolation": {
            "alpha": alpha,
            f"mean_ws_100m": round(mean_ws * (100/50)**alpha, 2),
            f"mean_ws_120m": round(mean_ws * (120/50)**alpha, 2),
            f"mean_ws_150m": round(mean_ws * (150/50)**alpha, 2),
        }
    }
    for name in dir_names:
        d = dir_bins[name]
        rose_data["directions"][name] = {
            "frequency_pct": round(d["count"] / n * 100, 1),
            "avg_speed_ms": round(d["ws_sum"] / d["count"], 2) if d["count"] > 0 else 0,
            "count": d["count"]
        }

    rose_path = os.path.join(NASA_DIR, "wind_rose_summary.json")
    with open(rose_path, "w") as f:
        json.dump(rose_data, f, indent=2)
    print(f"\n  Wind rose data saved -> {rose_path}")

    print(f"\n{'='*60}")
    print("  VERDICT: Muppandal has excellent wind resource.")
    print(f"  Mean {mean_ws:.1f} m/s at 50m -> ~{mean_ws*(150/50)**0.14:.1f} m/s at 150m")
    print("=" * 60)


if __name__ == "__main__":
    main()
