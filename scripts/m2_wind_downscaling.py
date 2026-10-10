"""
PawanSthal — Module M2: Wind Resource Downscaling Model
=======================================================
Downscales coarse reanalysis wind (NASA POWER 3x3 grid) to high-resolution
hub-height wind resource field (100m & 150m) across the Muppandal corridor.

Algorithm:
  - Input Features:
      * Interpolated coarse NASA POWER wind speed (WS50M, WS2M, T2M)
      * Spatial position: latitude, longitude, distance from Aralvaimozhi pass center
      * Geographic corridor coordinates & terrain gradient proxy
  - Ground Truth Labels:
      * High-res GWA raster if downloaded, or physics-calibrated power-law profile
  - Model:
      * HistGradientBoostingRegressor (scikit-learn LightGBM equivalent)
  - Output:
      * High-resolution wind speed grid JSON (for Leaflet heatmap visualization)
      * Model performance metrics (RMSE, MAE, R^2)
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config

GRID_META_FILE = os.path.join(config.NASA_DIR, "grid_metadata.json")
GWA_TIF_100M = os.path.join(config.GWA_DIR, "IND_wind-speed_100m.tif")
OUTPUT_HEATMAP_JSON = os.path.join(config.DATA_DIR, "wind_heatmap_grid.json")
OUTPUT_M2_SUMMARY = os.path.join(config.DATA_DIR, "m2_downscaling_summary.json")

def load_nasa_grid_points():
    """Load the 9 NASA POWER grid points with their annual mean wind speeds."""
    with open(GRID_META_FILE, "r") as f:
        meta = json.load(f)

    points = []
    for pt in meta.get("points", []):
        idx = pt["index"]
        lat = pt["lat"]
        lon = pt["lon"]
        csv_name = f"grid_point_{idx:02d}_{lat}_{lon}.csv"
        csv_path = os.path.join(config.NASA_DIR, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            ws50_mean = float(df["WS50M"].mean())
            ws2_mean = float(df["WS2M"].mean())
            t2m_mean = float(df["T2M"].mean())
            points.append({
                "lat": lat,
                "lon": lon,
                "ws50m": ws50_mean,
                "ws2m": ws2_mean,
                "t2m": t2m_mean
            })
    return pd.DataFrame(points)

def generate_spatial_dataset(nasa_df, n_samples=3000):
    """
    Generate dataset matching NASA POWER coarse reanalysis inputs to exact
    Global Wind Atlas ground truth labels.
    """
    with open(OUTPUT_HEATMAP_JSON, "r", encoding="utf-8") as f:
        gwa_data = json.load(f)

    gwa_points = gwa_data.get("points", [])
    lats = np.array([p["lat"] for p in gwa_points])
    lons = np.array([p["lon"] for p in gwa_points])
    y_100 = np.array([p["ws_100m"] for p in gwa_points])
    y_120 = np.array([p["ws_120m"] for p in gwa_points])
    y_150 = np.array([p["ws_150m"] for p in gwa_points])
    n_pts = len(lats)

    # Venturi corridor distance from Aralvaimozhi Pass (8.22 N, 77.53 E)
    pass_lat, pass_lon = 8.22, 77.53
    dist_pass_km = np.sqrt(((lats - pass_lat) * 111.0)**2 + ((lons - pass_lon) * 110.0)**2)

    # Inverse distance weighting from 9 NASA POWER grid points
    coords_nasa = nasa_df[["lat", "lon"]].values
    ws50_interp = np.zeros(n_pts)
    ws2_interp = np.zeros(n_pts)
    t2m_interp = np.zeros(n_pts)

    for i in range(n_pts):
        dists = np.sqrt((coords_nasa[:, 0] - lats[i])**2 + (coords_nasa[:, 1] - lons[i])**2)
        weights = 1.0 / (dists + 1e-5)**2
        weights /= weights.sum()
        ws50_interp[i] = np.sum(weights * nasa_df["ws50m"])
        ws2_interp[i] = np.sum(weights * nasa_df["ws2m"])
        t2m_interp[i] = np.sum(weights * nasa_df["t2m"])

    features_df = pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "dist_pass_km": dist_pass_km,
        "coarse_ws50m": ws50_interp,
        "coarse_ws2m": ws2_interp,
        "coarse_t2m": t2m_interp,
        "ws_100m": y_100,
        "ws_120m": y_120,
        "ws_150m": y_150
    })
    return features_df

def train_downscaling_model():
    print("=" * 60)
    print("PawanSthal — Module M2: Wind Resource Downscaling Regressor")
    print("=" * 60)

    nasa_df = load_nasa_grid_points()
    print(f"Loaded {len(nasa_df)} NASA POWER reanalysis grid points.")

    print("Generating feature matrix (spatial coordinates, Venturi corridor index, coarse reanalysis)...")
    dataset = generate_spatial_dataset(nasa_df, n_samples=4000)

    feature_cols = ["lat", "lon", "dist_pass_km", "coarse_ws50m", "coarse_ws2m", "coarse_t2m"]
    X = dataset[feature_cols]
    y_120 = dataset["ws_120m"]
    y_150 = dataset["ws_150m"]

    X_train, X_test, y_train, y_test = train_test_split(X, y_120, test_size=0.2, random_state=42)

    print("Training HistGradientBoostingRegressor (LightGBM equivalent)...")
    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Model Evaluation (120m Hub Height) ---")
    print(f"R^2 Score: {r2:.4f}")
    print(f"RMSE:      {rmse:.3f} m/s")
    print(f"MAE:       {mae:.3f} m/s")

    # Generate fine grid for UI heatmap (30x30 = 900 points across Muppandal)
    bbox = config.MUPPANDAL_BBOX
    grid_lats = np.linspace(bbox["south"], bbox["north"], 32)
    grid_lons = np.linspace(bbox["west"], bbox["east"], 32)

    grid_records = []
    coords_nasa = nasa_df[["lat", "lon"]].values

    for lat in grid_lats:
        for lon in grid_lons:
            dist_pass = math.sqrt(((lat - 8.22) * 111.0)**2 + ((lon - 77.53) * 110.0)**2)
            dists = np.sqrt((coords_nasa[:, 0] - lat)**2 + (coords_nasa[:, 1] - lon)**2)
            weights = 1.0 / (dists + 1e-5)**2
            weights /= weights.sum()
            c_ws50 = np.sum(weights * nasa_df["ws50m"])
            c_ws2 = np.sum(weights * nasa_df["ws2m"])
            c_t2m = np.sum(weights * nasa_df["t2m"])

            row_feats = pd.DataFrame([{
                "lat": lat, "lon": lon, "dist_pass_km": dist_pass,
                "coarse_ws50m": c_ws50, "coarse_ws2m": c_ws2, "coarse_t2m": c_t2m
            }])
            pred_120 = float(model.predict(row_feats)[0])
            pred_150 = round(pred_120 * ((150.0 / 120.0) ** 0.14), 2)
            pred_100 = round(pred_120 * ((100.0 / 120.0) ** 0.14), 2)

            grid_records.append({
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "ws_100m": pred_100,
                "ws_120m": round(pred_120, 2),
                "ws_150m": pred_150,
            })

    heatmap_payload = {
        "metadata": {
            "bbox": bbox,
            "grid_resolution": "32x32 (1024 cells, ~800m res)",
            "model": "HistGradientBoostingRegressor",
            "r2_score": round(r2, 4),
            "rmse_ms": round(rmse, 3),
            "min_ws_120m": min(g["ws_120m"] for g in grid_records),
            "max_ws_120m": max(g["ws_120m"] for g in grid_records),
            "mean_ws_120m": round(sum(g["ws_120m"] for g in grid_records) / len(grid_records), 2),
        },
        "points": grid_records
    }

    with open(OUTPUT_HEATMAP_JSON, "w", encoding="utf-8") as f:
        json.dump(heatmap_payload, f, indent=2)
    print(f"\nSaved wind resource heatmap grid to: {OUTPUT_HEATMAP_JSON}")

    summary = {
        "r2_score": round(r2, 4),
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "total_grid_points": len(grid_records),
        "min_ws_120m": heatmap_payload["metadata"]["min_ws_120m"],
        "max_ws_120m": heatmap_payload["metadata"]["max_ws_120m"],
        "mean_ws_120m": heatmap_payload["metadata"]["mean_ws_120m"]
    }
    with open(OUTPUT_M2_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved M2 summary to: {OUTPUT_M2_SUMMARY}")

if __name__ == "__main__":
    train_downscaling_model()
