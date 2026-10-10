"""
Fetch OSM data (turbines, substations, roads, buildings) for Muppandal cluster.
Uses the Overpass API - no API key required.

Saves GeoJSON files to data/osm/
"""

import os
import sys
import json
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MUPPANDAL_BBOX, OSM_DIR

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def overpass_query(query, retries=3):
    """Execute an Overpass QL query and return the JSON result, with retries."""
    from urllib.parse import quote
    encoded_data = ("data=" + query).encode("utf-8")

    for attempt in range(retries):
        req = Request(
            OVERPASS_URL,
            data=encoded_data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "PawanSthal/1.0"
            },
            method="POST"
        )

        try:
            with urlopen(req, timeout=180) as response:
                return json.loads(response.read().decode())
        except HTTPError as e:
            print(f"  HTTP Error {e.code}: {e.reason} (attempt {attempt+1}/{retries})")
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
        except URLError as e:
            print(f"  URL Error: {e.reason}")
            if attempt < retries - 1:
                time.sleep(10)

    return None


def osm_to_geojson(osm_data, feature_type="Point"):
    """Convert Overpass JSON response to GeoJSON FeatureCollection."""
    features = []

    if not osm_data or "elements" not in osm_data:
        return {"type": "FeatureCollection", "features": []}

    for elem in osm_data["elements"]:
        if elem["type"] == "node" and "lat" in elem and "lon" in elem:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [elem["lon"], elem["lat"]]
                },
                "properties": {
                    "osm_id": elem["id"],
                    "osm_type": "node",
                    **elem.get("tags", {})
                }
            }
            features.append(feature)

        elif elem["type"] == "way" and "center" in elem:
            # For ways, use center point
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [elem["center"]["lon"], elem["center"]["lat"]]
                },
                "properties": {
                    "osm_id": elem["id"],
                    "osm_type": "way",
                    **elem.get("tags", {})
                }
            }
            features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def fetch_turbines(bbox):
    """Fetch wind turbines from OSM."""
    s, n, w, e = bbox["south"], bbox["north"], bbox["west"], bbox["east"]

    query = f"""
    [out:json][timeout:60];
    (
      node["generator:source"="wind"]({s},{w},{n},{e});
      way["generator:source"="wind"]({s},{w},{n},{e});
      node["power"="generator"]["generator:source"="wind"]({s},{w},{n},{e});
      way["power"="generator"]["generator:source"="wind"]({s},{w},{n},{e});
      node["man_made"="wind_turbine"]({s},{w},{n},{e});
      node["power"="generator"]["generator:type"="wind_turbine"]({s},{w},{n},{e});
    );
    out center body;
    """
    print("  Querying wind turbines...")
    return overpass_query(query)


def fetch_substations(bbox):
    """Fetch electrical substations from OSM."""
    s, n, w, e = bbox["south"], bbox["north"], bbox["west"], bbox["east"]

    query = f"""
    [out:json][timeout:60];
    (
      node["power"="substation"]({s},{w},{n},{e});
      way["power"="substation"]({s},{w},{n},{e});
      node["power"="sub_station"]({s},{w},{n},{e});
      way["power"="sub_station"]({s},{w},{n},{e});
    );
    out center body;
    """
    print("  Querying substations...")
    return overpass_query(query)


def fetch_roads(bbox):
    """Fetch major roads from OSM (for setback constraints)."""
    s, n, w, e = bbox["south"], bbox["north"], bbox["west"], bbox["east"]

    query = f"""
    [out:json][timeout:60];
    (
      way["highway"~"primary|secondary|tertiary|trunk|motorway"]({s},{w},{n},{e});
    );
    out center body;
    """
    print("  Querying roads...")
    return overpass_query(query)


def fetch_buildings(bbox):
    """Fetch buildings/settlements from OSM (for setback constraints)."""
    s, n, w, e = bbox["south"], bbox["north"], bbox["west"], bbox["east"]

    # Only fetch residential areas and named places, not individual buildings
    # (too many individual buildings would overload the query)
    query = f"""
    [out:json][timeout:60];
    (
      node["place"~"village|town|hamlet|locality"]({s},{w},{n},{e});
      way["landuse"="residential"]({s},{w},{n},{e});
    );
    out center body;
    """
    print("  Querying settlements & buildings...")
    return overpass_query(query)


def fetch_power_lines(bbox):
    """Fetch power lines from OSM."""
    s, n, w, e = bbox["south"], bbox["north"], bbox["west"], bbox["east"]

    query = f"""
    [out:json][timeout:60];
    (
      way["power"="line"]({s},{w},{n},{e});
      way["power"="minor_line"]({s},{w},{n},{e});
    );
    out center body;
    """
    print("  Querying power lines...")
    return overpass_query(query)


def save_geojson(geojson, filename):
    """Save GeoJSON to file."""
    path = os.path.join(OSM_DIR, filename)
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
    count = len(geojson.get("features", []))
    print(f"  Saved {count} features -> {path}")
    return count


def main():
    os.makedirs(OSM_DIR, exist_ok=True)
    bbox = MUPPANDAL_BBOX

    print("=" * 60)
    print("PawanSthal — OSM Data Fetcher (Overpass API)")
    print("=" * 60)
    print(f"Cluster: {bbox['name']}")
    print(f"Bbox: ({bbox['south']}, {bbox['west']}) – ({bbox['north']}, {bbox['east']})")
    print()

    summary = {}

    # 1. Wind turbines
    print("[1/5] Wind Turbines")
    turbine_data = fetch_turbines(bbox)
    turbine_geojson = osm_to_geojson(turbine_data)
    summary["turbines"] = save_geojson(turbine_geojson, "turbines.geojson")

    if summary["turbines"] < 30:
        print(f"  ⚠️  Only {summary['turbines']} turbines found in OSM.")
        print("  → You'll need to hand-digitise more in geojson.io from satellite view.")
        print("  → Target: 30–50 turbines for a good demo.")
    print()

    # 2. Substations
    print("[2/5] Substations")
    sub_data = fetch_substations(bbox)
    sub_geojson = osm_to_geojson(sub_data)
    summary["substations"] = save_geojson(sub_geojson, "substations.geojson")
    print()

    # 3. Roads
    print("[3/5] Roads")
    road_data = fetch_roads(bbox)
    road_geojson = osm_to_geojson(road_data)
    summary["roads"] = save_geojson(road_geojson, "roads.geojson")
    print()

    # 4. Settlements
    print("[4/5] Settlements & Buildings")
    building_data = fetch_buildings(bbox)
    building_geojson = osm_to_geojson(building_data)
    summary["settlements"] = save_geojson(building_geojson, "settlements.geojson")
    print()

    # 5. Power lines
    print("[5/5] Power Lines")
    power_data = fetch_power_lines(bbox)
    power_geojson = osm_to_geojson(power_data)
    summary["power_lines"] = save_geojson(power_geojson, "power_lines.geojson")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for layer, count in summary.items():
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {layer}: {count} features")

    print(f"\nAll data saved to: {OSM_DIR}")

    if summary["turbines"] < 30:
        print(f"\n⚠️  ACTION NEEDED:")
        print(f"  Open https://geojson.io and:")
        print(f"  1. Navigate to Muppandal ({bbox['center_lat']}, {bbox['center_lon']})")
        print(f"  2. Switch to satellite view")
        print(f"  3. Click on turbine locations to add points")
        print(f"  4. Export as GeoJSON → save to data/turbines/manual_turbines.geojson")

    print("=" * 60)


if __name__ == "__main__":
    main()
