"""
PawanSthal — Crop Global Wind Atlas (GWA) GeoTIFF for Muppandal Complex
=======================================================================
Extracts exact 250m ground truth wind speed data for Muppandal bounding box:
  - Lat: 8.08° to 8.30° N
  - Lon: 77.34° to 77.60° E
Georeferencing:
  - Origin: Lon 65.61875°, Lat 33.27875°
  - Pixel size: 0.0025° x 0.0025° (WGS 84)
"""

import os
import sys
import json
import numpy as np
import tifffile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config

GWA_100M_TIF = os.path.join(config.GWA_DIR, "IND_wind-speed_100m.tif")
OUTPUT_CROPPED_TIF = os.path.join(config.GWA_DIR, "muppandal_gwa_100m_cropped.tif")
OUTPUT_CROPPED_JSON = os.path.join(config.GWA_DIR, "muppandal_gwa_100m.json")

def crop_muppandal():
    print("=" * 60)
    print("Cropping Global Wind Atlas 100m Raster for Muppandal")
    print("=" * 60)

    if not os.path.exists(GWA_100M_TIF):
        print(f"Error: {GWA_100M_TIF} not found!")
        return

    print("Reading full India GWA GeoTIFF...")
    with tifffile.TiffFile(GWA_100M_TIF) as tif:
        page = tif.pages[0]
        # Full shape: (11404, 12627)
        # Georeferencing parameters
        origin_lon = 65.61875
        origin_lat = 33.27875
        pixel_size = 0.0025

        bbox = config.MUPPANDAL_BBOX
        # Row calculation (lat decreases down rows)
        row_min = int(round((origin_lat - bbox["north"]) / pixel_size))
        row_max = int(round((origin_lat - bbox["south"]) / pixel_size))
        # Col calculation (lon increases across cols)
        col_min = int(round((bbox["west"] - origin_lon) / pixel_size))
        col_max = int(round((bbox["east"] - origin_lon) / pixel_size))

        print(f"BBox: Lat [{bbox['south']}, {bbox['north']}], Lon [{bbox['west']}, {bbox['east']}]")
        print(f"Pixel slice: Rows [{row_min}:{row_max}], Cols [{col_min}:{col_max}]")

        # Read only the needed block or full image
        data = page.asarray()
        cropped = data[row_min:row_max, col_min:col_max]

    print(f"Cropped shape: {cropped.shape} ({cropped.shape[0]} rows x {cropped.shape[1]} cols)")
    
    # Filter valid values (GWA uses -9999 or NaN for nodata/water)
    valid_mask = (cropped > 0) & (cropped < 50) & ~np.isnan(cropped)
    valid_data = cropped[valid_mask]

    min_val = float(np.min(valid_data))
    max_val = float(np.max(valid_data))
    mean_val = float(np.mean(valid_data))
    median_val = float(np.median(valid_data))

    print(f"\n--- GWA Ground Truth Wind Speed Statistics (100m Height) ---")
    print(f"Min Wind Speed:    {min_val:.2f} m/s")
    print(f"Max Wind Speed:    {max_val:.2f} m/s (peak in Aralvaimozhi Pass!)")
    print(f"Mean Wind Speed:   {mean_val:.2f} m/s")
    print(f"Median Wind Speed: {median_val:.2f} m/s")

    # Save cropped GeoTIFF
    tifffile.imwrite(
        OUTPUT_CROPPED_TIF,
        cropped.astype(np.float32),
        photometric='minisblack'
    )
    print(f"\nSaved cropped GeoTIFF to: {OUTPUT_CROPPED_TIF}")

    # Build geo-located point cloud JSON for frontend/analysis
    rows, cols = cropped.shape
    point_records = []
    step = 2  # sample every 2nd pixel for compact web payload (approx 500m res)
    for r in range(0, rows, step):
        lat = origin_lat - (row_min + r) * pixel_size
        for c in range(0, cols, step):
            lon = origin_lon + (col_min + c) * pixel_size
            ws = float(cropped[r, c])
            if ws > 0 and ws < 50 and not np.isnan(ws):
                ws_120 = round(ws * ((120.0 / 100.0) ** 0.14), 2)
                ws_150 = round(ws * ((150.0 / 100.0) ** 0.14), 2)
                point_records.append({
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "ws_100m": round(ws, 2),
                    "ws_120m": ws_120,
                    "ws_150m": ws_150
                })

    meta_dict = {
        "source": "Global Wind Atlas (DTU / World Bank) Version 4",
        "bbox": bbox,
        "grid_points_count": len(point_records),
        "height_m": 100,
        "stats_100m": {
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "mean": round(mean_val, 2),
            "median": round(median_val, 2)
        },
        "points": point_records
    }

    with open(OUTPUT_CROPPED_JSON, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"Saved cropped GWA JSON to: {OUTPUT_CROPPED_JSON}")

    # Also update wind_heatmap_grid.json with real GWA data!
    heatmap_path = os.path.join(config.DATA_DIR, "wind_heatmap_grid.json")
    with open(heatmap_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"Updated live web heatmap grid with real GWA data!")

if __name__ == "__main__":
    crop_muppandal()
