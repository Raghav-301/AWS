"""
PawanSthal — Module M1: Random Forest Turbine Classifier Training
=================================================================
Trains a Random Forest classifier on spatial features (k-NN distance,
local density, distance to road & substation) to classify turbines into:
  - 'old' (<1 MW, e.g., 250-500 kW, tightly packed)
  - 'mid' (1-2 MW, 70-80m rotor, moderate spacing)
  - 'modern' (>2 MW, 110-140m rotor, wide spacing)

Outputs:
  - Train vs Validation Loss (Cross-Entropy), Accuracy, Precision/Recall
  - Feature Importance Ranking
  - Model serialization to data/models/m1_turbine_rf.joblib
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, classification_report
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config
from scripts.m1_turbine_classifier import (
    load_geojson_coords,
    extract_spatial_features,
    classify_turbines,
    TURBINES_GEOJSON
)

MODELS_DIR = os.path.join(config.DATA_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

OUTPUT_RF_MODEL = os.path.join(MODELS_DIR, "m1_turbine_rf.joblib")
OUTPUT_RF_METRICS = os.path.join(config.DATA_DIR, "m1_rf_metrics.json")


def train_m1_classifier():
    print("=" * 70)
    print(" PawanSthal — Module M1: Random Forest Classifier Training ")
    print("=" * 70)

    with open(TURBINES_GEOJSON, "r", encoding="utf-8") as f:
        turbines_data = json.load(f)

    raw_features = turbines_data.get("features", [])
    print(f"Loaded {len(raw_features)} turbines from OpenStreetMap.")

    turbines = []
    for idx, f in enumerate(raw_features):
        coords = f["geometry"]["coordinates"]
        turbines.append({
            "id": f.get("properties", {}).get("osm_id", f"turb_{idx}"),
            "lon": coords[0],
            "lat": coords[1],
            "properties": f.get("properties", {}),
            "geometry": f["geometry"]
        })

    print("Computing spatial features (k-NN, local densities, road/substation distances)...")
    df_features = extract_spatial_features(turbines)
    df_classified = classify_turbines(df_features)

    feature_cols = [
        "nn1_dist_m", "nn2_dist_m", "nn3_dist_m", "nn5_dist_m",
        "mean_nn3_m", "density_300m", "density_500m", "density_1000m",
        "dist_substation_m", "dist_road_m"
    ]

    X = np.asarray(df_classified[feature_cols].values, dtype=np.float64)
    y = np.asarray(df_classified["turbine_class"].tolist(), dtype=object)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\nDataset Partition:")
    print(f"  - Training samples:   {len(X_train)} (80%)")
    print(f"  - Validation samples: {len(X_val)} (20%)")
    print(f"  - Features: {feature_cols}")

    print("\nTraining RandomForestClassifier (n_estimators=100, max_depth=8)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, oob_score=True)
    rf.fit(X_train, y_train)

    # Probabilities for log-loss
    train_probs = rf.predict_proba(X_train)
    val_probs = rf.predict_proba(X_val)

    train_loss = log_loss(y_train, train_probs, labels=rf.classes_)
    val_loss = log_loss(y_val, val_probs, labels=rf.classes_)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc = accuracy_score(y_val, rf.predict(X_val))

    print("\n--- Training & Validation Metrics ---")
    print(f"  • Train Cross-Entropy Loss: {train_loss:.4f}  |  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  • Val Cross-Entropy Loss:   {val_loss:.4f}  |  Val Accuracy:   {val_acc*100:.2f}%")
    print(f"  • Out-of-Bag (OOB) Score:   {rf.oob_score_*100:.2f}%")

    print("\n--- Detailed Validation Report ---")
    val_preds = rf.predict(X_val)
    print(classification_report(y_val, val_preds, target_names=rf.classes_))

    print("--- Feature Importances ---")
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_imp_dict = {}
    for i in sorted_idx:
        feat = feature_cols[i]
        imp = importances[i]
        feature_imp_dict[feat] = round(float(imp), 4)
        print(f"  • {feat:<20}: {imp*100:5.2f}%")

    package = {
        "model": rf,
        "feature_cols": feature_cols,
        "classes": list(rf.classes_),
        "metrics": {
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "oob_score": round(rf.oob_score_, 4),
            "feature_importances": feature_imp_dict
        }
    }

    joblib.dump(package, OUTPUT_RF_MODEL)
    print(f"\nSaved Random Forest model to: {OUTPUT_RF_MODEL}")

    with open(OUTPUT_RF_METRICS, "w", encoding="utf-8") as f:
        json.dump(package["metrics"], f, indent=2)
    print(f"Saved metrics summary to: {OUTPUT_RF_METRICS}")
    print("=" * 70)


if __name__ == "__main__":
    train_m1_classifier()
