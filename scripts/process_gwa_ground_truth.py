"""
PawanSthal — Process GWA Ground Truth (100m & 150m) for Muppandal
================================================================
Decodes Global Wind Atlas GeoTIFFs (100m & 150m) using pure Python + zstandard +
floating point predictor inversion. Converts km/h to m/s and saves ground truth
wind resource grid for Module M2 retraining and the live decision dashboard.
"""

import os
import sys
import json
import numpy as np
import tifffile
import zstandard

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config

GWA_100M_PATH = os.path.join(config.GWA_DIR, "IND_wind-speed_100m.tif")
GWA_150M_PATH = os.path.join(config.GWA_DIR, "IND_wind-speed_150m.tif")
OUTPUT_HEATMAP_JSON = os.path.join(config.DATA_DIR, "wind_heatmap_grid.json")

def decode_gwa_tile(tif_path, tile_idx=484):
    """Decode a single 512x512 tile from GWA GeoTIFF (zstd + predictor 3)."""
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        offset = page.dataoffsets[tile_idx]
        bytecount = page.databytecounts[tile_idx]

    with open(tif_path, 'rb') as f:
        f.seek(offset)
        comp = f.read(bytecount)

    dctx = zstandard.ZstdDecompressor()
    raw = dctx.decompress(comp, max_output_size=512 * 512 * 4)
    raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape((512, 4, 512))
    unpred = np.cumsum(raw_arr, axis=2, dtype=np.uint8)
    # Transpose and view as big-endian float32
    tile_float = np.transpose(unpred, (0, 2, 1)).copy().view('>f4').reshape((512, 512))
    # Convert from km/h to m/s
    tile_ms = tile_float / 3.6
    return tile_ms

def extract_muppandal_grid():
    print("=" * 60)
    print("Processing Real GWA Ground Truth (100m & 150m) for Muppandal")
    print("=" * 60)

    tile_100m = decode_gwa_tile(GWA_100M_PATH, tile_idx=484)
    tile_150m = decode_gwa_tile(GWA_150M_PATH, tile_idx=484)

    origin_lon = 65.61875
    origin_lat = 33.27875
    pixel_size = 0.0025

    bbox = config.MUPPANDAL_BBOX
    # Pixel slice covering Muppandal
    row_min = 263
    row_max = 352
    col_min = 80
    col_max = 184

    mup_100m = tile_100m[row_min:row_max, col_min:col_max]
    mup_150m = tile_150m[row_min:row_max, col_min:col_max]

    print(f"Extracted Muppandal grid: {mup_100m.shape} pixels (250m resolution)")
    print(f"100m Wind Speed: Min={mup_100m.min():.2f} m/s, Max={mup_100m.max():.2f} m/s, Mean={mup_100m.mean():.2f} m/s")
    print(f"150m Wind Speed: Min={mup_150m.min():.2f} m/s, Max={mup_150m.max():.2f} m/s, Mean={mup_150m.mean():.2f} m/s")

    # Build point cloud for web heatmap
    points = []
    rows, cols = mup_100m.shape
    step = 2  # ~500m resolution for smooth web rendering
    for r in range(0, rows, step):
        lat = origin_lat - (19 * 512 + row_min + r) * pixel_size
        for c in range(0, cols, step):
            lon = origin_lon + (9 * 512 + col_min + c) * pixel_size
            ws_100 = float(mup_100m[r, c])
            ws_150 = float(mup_150m[r, c])
            ws_120 = round(ws_100 * ((120.0 / 100.0) ** 0.14), 2)

            points.append({
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "ws_100m": round(ws_100, 2),
                "ws_120m": ws_120,
                "ws_150m": round(ws_150, 2)
            })

    heatmap_payload = {
        "metadata": {
            "source": "Global Wind Atlas (DTU Wind Energy / World Bank) Version 4",
            "bbox": bbox,
            "total_points": len(points),
            "stats_100m": {
                "min": round(float(mup_100m.min()), 2),
                "max": round(float(mup_100m.max()), 2),
                "mean": round(float(mup_100m.mean()), 2)
            },
            "stats_150m": {
                "min": round(float(mup_150m.min()), 2),
                "max": round(float(mup_150m.max()), 2),
                "mean": round(float(mup_150m.mean()), 2)
            }
        },
        "points": points
    }

    with open(OUTPUT_HEATMAP_JSON, "w", encoding="utf-8") as f:
        json.dump(heatmap_payload, f, indent=2)
    print(f"\nSaved real GWA heatmap grid to: {OUTPUT_HEATMAP_JSON}")

if __name__ == "__main__":
    extract_muppandal_grid()
