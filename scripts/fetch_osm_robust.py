"""
Fetch OSM data using requests library (more robust than urllib).
Queries Overpass API for turbines, substations, roads, settlements, power lines.
"""

import os
import sys
import json
import time
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MUPPANDAL_BBOX, OSM_DIR

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Fallback mirrors if main server is overloaded
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def run_query(query, retries=3):
    """Run Overpass query with retries and mirror fallback."""
    for mirror in OVERPASS_MIRRORS:
        for attempt in range(retries):
            try:
                print(f"    Trying {mirror.split('/')[2]} (attempt {attempt+1})...")
                resp = requests.post(
                    mirror,
                    data={"data": query},
                    timeout=180,
                    headers={"User-Agent": "PawanSthal/1.0"}
                )
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    print(f"    Rate limited, waiting 30s...")
                    time.sleep(30)
                elif resp.status_code == 504:
                    print(f"    Gateway timeout, waiting 15s...")
                    time.sleep(15)
                else:
                    print(f"    HTTP {resp.status_code}, waiting 10s...")
                    time.sleep(10)
            except requests.exceptions.Timeout:
                print(f"    Request timeout, trying next...")
                time.sleep(5)
            except Exception as e:
                print(f"    Error: {e}")
                time.sleep(5)
    return None


def to_geojson(osm_data):
    """Convert Overpass JSON to GeoJSON."""
    features = []
    if not osm_data or "elements" not in osm_data:
        return {"type": "FeatureCollection", "features": []}

    for el in osm_data["elements"]:
        coords = None
        if el["type"] == "node" and "lat" in el and "lon" in el:
            coords = [el["lon"], el["lat"]]
        elif el["type"] == "way" and "center" in el:
            coords = [el["center"]["lon"], el["center"]["lat"]]

        if coords:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": coords},
                "properties": {"osm_id": el["id"], "osm_type": el["type"], **el.get("tags", {})}
            })

    return {"type": "FeatureCollection", "features": features}


def save(geojson, name):
    """Save GeoJSON to disk."""
    path = os.path.join(OSM_DIR, name)
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
    n = len(geojson.get("features", []))
    print(f"    -> {n} features saved to {name}")
    return n


# ---- Layer queries ----

LAYERS = {
    "turbines": {
        "file": "turbines.geojson",
        "query": """
[out:json][timeout:120];
(
  node["generator:source"="wind"]({s},{w},{n},{e});
  way["generator:source"="wind"]({s},{w},{n},{e});
  node["man_made"="wind_turbine"]({s},{w},{n},{e});
  node["power"="generator"]["generator:source"="wind"]({s},{w},{n},{e});
  way["power"="generator"]["generator:source"="wind"]({s},{w},{n},{e});
);
out center body;
"""
    },
    "substations": {
        "file": "substations.geojson",
        "query": """
[out:json][timeout:60];
(
  node["power"="substation"]({s},{w},{n},{e});
  way["power"="substation"]({s},{w},{n},{e});
);
out center body;
"""
    },
    "roads": {
        "file": "roads.geojson",
        "query": """
[out:json][timeout:60];
way["highway"~"primary|secondary|tertiary|trunk|motorway"]({s},{w},{n},{e});
out center body;
"""
    },
    "settlements": {
        "file": "settlements.geojson",
        "query": """
[out:json][timeout:60];
(
  node["place"~"village|town|hamlet|locality"]({s},{w},{n},{e});
);
out center body;
"""
    },
    "power_lines": {
        "file": "power_lines.geojson",
        "query": """
[out:json][timeout:60];
way["power"="line"]({s},{w},{n},{e});
out center body;
"""
    },
}


def main():
    os.makedirs(OSM_DIR, exist_ok=True)
    bbox = MUPPANDAL_BBOX
    fmt = {"s": bbox["south"], "n": bbox["north"], "w": bbox["west"], "e": bbox["east"]}

    print("=" * 60)
    print("PawanSthal - OSM Data Fetcher")
    print("=" * 60)
    print(f"Cluster: {bbox['name']}")
    print(f"Bbox: ({bbox['south']}, {bbox['west']}) - ({bbox['north']}, {bbox['east']})")
    print()

    summary = {}
    for i, (name, layer) in enumerate(LAYERS.items()):
        outpath = os.path.join(OSM_DIR, layer["file"])

        # Skip if already downloaded
        if os.path.exists(outpath) and os.path.getsize(outpath) > 50:
            with open(outpath) as f:
                existing = json.load(f)
            n = len(existing.get("features", []))
            if n > 0:
                print(f"  [{i+1}/{len(LAYERS)}] {name}: SKIP ({n} features already saved)")
                summary[name] = n
                continue

        print(f"  [{i+1}/{len(LAYERS)}] {name}:")
        query = layer["query"].format(**fmt)
        result = run_query(query)
        geojson = to_geojson(result)
        summary[name] = save(geojson, layer["file"])
        time.sleep(3)  # politeness delay

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for name, count in summary.items():
        print(f"  {'[OK]' if count > 0 else '[!!]'} {name}: {count} features")

    turbines = summary.get("turbines", 0)
    if turbines < 30:
        print(f"\n  [ACTION NEEDED] Only {turbines} turbines in OSM.")
        print(f"  Hand-digitise more at: https://geojson.io")
        print(f"  Navigate to ({bbox['center_lat']}, {bbox['center_lon']})")
        print(f"  Target: 30-50 turbines, save as data/turbines/manual_turbines.geojson")

    print("=" * 60)


if __name__ == "__main__":
    main()
