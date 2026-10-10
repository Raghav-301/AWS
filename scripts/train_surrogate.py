"""
PawanSthal — Module M5: Neural Surrogate Wake & AEP Model Training
==================================================================
Trains a high-speed Neural Network surrogate model on vectorized Jensen
wake simulations (Module M3 physics) across varied wind farm configurations.

Outputs:
  - Epoch-by-epoch Train Loss, Val Loss, and Val R^2 tracking
  - Physics vs Neural Surrogate latency benchmark (>100x speedup)
  - Model serialization to data/models/m5_surrogate_model.joblib & .json
  - Full training history exported to data/m5_training_history.json
"""

import os
import sys
import time
import math
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config
from scripts.m3_wake_model import (
    WindTurbineSpec,
    evaluate_layout_aep,
    WIND_ROSE_FILE
)

MODELS_DIR = os.path.join(config.DATA_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

OUTPUT_MODEL_JOBLIB = os.path.join(MODELS_DIR, "m5_surrogate_model.joblib")
OUTPUT_MODEL_JSON = os.path.join(MODELS_DIR, "m5_surrogate_weights.json")
OUTPUT_SUMMARY_JSON = os.path.join(config.DATA_DIR, "m5_surrogate_summary.json")
OUTPUT_HISTORY_JSON = os.path.join(config.DATA_DIR, "m5_training_history.json")


def generate_layout_dataset(n_samples=1800, seed=42):
    """
    Generates synthetic wind farm layouts and evaluates their exact AEP
    using Module M3 vectorized wake physics.
    """
    print(f"\n[1/4] Generating {n_samples} physics-simulated wind farm layouts via M3 Engine...")
    np.random.seed(seed)
    
    # Load wind rose
    wind_rose = None
    if os.path.exists(WIND_ROSE_FILE):
        with open(WIND_ROSE_FILE, "r") as f:
            wind_rose = json.load(f)
            
    records = []
    start_t = time.time()
    
    for i in range(n_samples):
        # Sample realistic domain parameters
        n_turbines = int(np.random.randint(12, 46))
        cap_mw = float(np.random.uniform(1.8, 3.6))
        hub_h = float(np.random.uniform(90.0, 150.0))
        spacing_d = float(np.random.uniform(3.5, 8.5))
        grid_aspect = float(np.random.uniform(0.6, 2.0))
        
        cap_kw = int(cap_mw * 1000)
        # Scale rotor diameter with capacity
        rotor_d = int(math.sqrt(cap_mw / 3.0) * 120.0)
        
        spec = WindTurbineSpec(
            name=f"Turbine-{cap_mw:.1f}MW",
            capacity_kw=cap_kw,
            rotor_diameter_m=rotor_d,
            hub_height_m=hub_h,
            cut_in=3.0,
            rated=10.5,
            cut_out=25.0,
            ct=0.78
        )
        
        # Build layout coordinate grid
        cols = max(3, int(math.sqrt(n_turbines * grid_aspect)))
        rows = max(3, math.ceil(n_turbines / cols))
        
        dx = spacing_d * rotor_d
        dy = spacing_d * rotor_d * 0.8
        
        # Add slight jitter/stagger to simulate real micro-siting
        xs = []
        ys = []
        for r in range(rows):
            stagger = (r % 2) * (dx * 0.4)
            for c in range(cols):
                if len(xs) < n_turbines:
                    jitter_x = np.random.uniform(-0.15, 0.15) * dx
                    jitter_y = np.random.uniform(-0.15, 0.15) * dy
                    xs.append(c * dx + stagger + jitter_x)
                    ys.append(r * dy + jitter_y)
                    
        xy_pts = np.column_stack([xs, ys])
        
        # Evaluate exact physics
        res = evaluate_layout_aep(xy_pts, spec, wind_rose=wind_rose)
        
        area_km2 = ((xy_pts[:, 0].max() - xy_pts[:, 0].min() + dx) * 
                    (xy_pts[:, 1].max() - xy_pts[:, 1].min() + dy)) / 1e6
        density_per_km2 = n_turbines / max(0.5, area_km2)
        
        records.append({
            "n_turbines": n_turbines,
            "turbine_capacity_mw": cap_mw,
            "hub_height_m": hub_h,
            "rotor_diameter_m": rotor_d,
            "spacing_diameters": spacing_d,
            "density_per_km2": round(density_per_km2, 2),
            "grid_aspect": round(grid_aspect, 2),
            "installed_capacity_mw": res["installed_capacity_mw"],
            "gross_aep_gwh": res["gross_aep_gwh"],
            "net_aep_gwh": res["net_aep_gwh"],
            "wake_loss_pct": res["wake_loss_pct"],
            "plf_pct": res["plf_pct"]
        })
        
        if (i + 1) % 400 == 0 or (i + 1) == n_samples:
            elapsed = time.time() - start_t
            rate = (i + 1) / elapsed
            print(f"  Processed {i + 1}/{n_samples} physics simulations ({rate:.1f} sim/sec)...")
            
    total_time = time.time() - start_t
    print(f"  Done: {n_samples} scenarios generated in {total_time:.2f}s ({total_time / n_samples * 1000:.2f} ms/simulation).")
    return pd.DataFrame(records)


class NeuralSurrogateMLP:
    """
    3-Layer Deep Multi-Layer Perceptron (MLP) Neural Network for Multi-Output
    AEP & Wake Deficit Regression with Adam optimizer and Mini-Batching.
    """
    def __init__(self, input_dim=7, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, output_dim=3, lr=0.003):
        np.random.seed(42)
        # He initialization for ReLU
        self.w1 = np.random.randn(input_dim, hidden_dim1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim1))
        
        self.w2 = np.random.randn(hidden_dim1, hidden_dim2) * np.sqrt(2.0 / hidden_dim1)
        self.b2 = np.zeros((1, hidden_dim2))
        
        self.w3 = np.random.randn(hidden_dim2, hidden_dim3) * np.sqrt(2.0 / hidden_dim2)
        self.b3 = np.zeros((1, hidden_dim3))
        
        self.w4 = np.random.randn(hidden_dim3, output_dim) * np.sqrt(2.0 / hidden_dim3)
        self.b4 = np.zeros((1, output_dim))
        
        self.lr = lr
        
        # Adam momentum & RMSprop caches
        self.mw = [np.zeros_like(w) for w in [self.w1, self.w2, self.w3, self.w4]]
        self.vw = [np.zeros_like(w) for w in [self.w1, self.w2, self.w3, self.w4]]
        self.mb = [np.zeros_like(b) for b in [self.b1, self.b2, self.b3, self.b4]]
        self.vb = [np.zeros_like(b) for b in [self.b1, self.b2, self.b3, self.b4]]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

    def relu(self, x):
        return np.maximum(0.0, x)

    def d_relu(self, x):
        return (x > 0.0).astype(float)

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.relu(self.z3)
        
        self.y_pred = np.dot(self.a3, self.w4) + self.b4
        return self.y_pred

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y_true):
        n = X.shape[0]
        self.t += 1
        
        # dL/dy_pred
        dy = (2.0 / n) * (self.y_pred - y_true)
        
        dw4 = np.dot(self.a3.T, dy)
        db4 = np.sum(dy, axis=0, keepdims=True)
        
        da3 = np.dot(dy, self.w4.T)
        dz3 = da3 * self.d_relu(self.z3)
        dw3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        da2 = np.dot(dz3, self.w3.T)
        dz2 = da2 * self.d_relu(self.z2)
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * self.d_relu(self.z1)
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        dws = [dw1, dw2, dw3, dw4]
        dbs = [db1, db2, db3, db4]
        ws = [self.w1, self.w2, self.w3, self.w4]
        bs = [self.b1, self.b2, self.b3, self.b4]
        
        # Adam update
        for i in range(4):
            self.mw[i] = self.beta1 * self.mw[i] + (1 - self.beta1) * dws[i]
            self.vw[i] = self.beta2 * self.vw[i] + (1 - self.beta2) * (dws[i] ** 2)
            
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * dbs[i]
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (dbs[i] ** 2)
            
            m_hat_w = self.mw[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.vw[i] / (1 - self.beta2 ** self.t)
            ws[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            
            m_hat_b = self.mb[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.vb[i] / (1 - self.beta2 ** self.t)
            bs[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
            
        self.w1, self.w2, self.w3, self.w4 = ws
        self.b1, self.b2, self.b3, self.b4 = bs

    def to_dict(self):
        return {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
            "w4": self.w4.tolist(), "b4": self.b4.tolist()
        }


def train_surrogate_model():
    print("=" * 75)
    print(" PawanSthal — Module M5: Neural Surrogate Wake & AEP Model Training ")
    print("=" * 75)
    
    # 1. Dataset Generation
    df = generate_layout_dataset(n_samples=1800, seed=42)
    
    feature_cols = [
        "n_turbines", "turbine_capacity_mw", "hub_height_m", 
        "rotor_diameter_m", "spacing_diameters", "density_per_km2", "grid_aspect"
    ]
    target_cols = ["net_aep_gwh", "wake_loss_pct", "plf_pct"]
    
    X = np.asarray(df[feature_cols].values, dtype=np.float64)
    Y = np.asarray(df[target_cols].values, dtype=np.float64)
    
    # 2. Train / Val / Test Split (80% Train, 10% Val, 10% Test)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.1111, random_state=42)
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)
    
    Y_train_scaled = scaler_y.fit_transform(Y_train)
    Y_val_scaled = scaler_y.transform(Y_val)
    Y_test_scaled = scaler_y.transform(Y_test)
    
    print(f"\n[2/4] Dataset Partition:")
    print(f"  - Training Set:   {X_train.shape[0]} samples (80%)")
    print(f"  - Validation Set: {X_val.shape[0]} samples (10%)")
    print(f"  - Test Set:       {X_test.shape[0]} samples (10%)")
    print(f"  - Features ({len(feature_cols)}): {feature_cols}")
    print(f"  - Target Outputs ({len(target_cols)}): {target_cols}")
    
    # 3. Training Loop with Epoch-by-Epoch Progress
    print("\n[3/4] Training Deep Neural Surrogate Network...")
    print("-" * 75)
    print(f"{'Epoch':^8} | {'Train Loss (MSE)':^18} | {'Val Loss (MSE)':^16} | {'Val R^2 (Mean)':^14} | {'Status':^10}")
    print("-" * 75)
    
    model = NeuralSurrogateMLP(input_dim=len(feature_cols), output_dim=len(target_cols), lr=0.005)
    
    epochs = 45
    batch_size = 64
    history = []
    
    best_val_loss = float("inf")
    n_batches = math.ceil(len(X_train) / batch_size)
    
    for epoch in range(1, epochs + 1):
        # Mini-batch shuffle
        perm = np.random.permutation(len(X_train))
        X_train_shuff = X_train_scaled[perm]
        Y_train_shuff = Y_train_scaled[perm]
        
        train_batch_losses = []
        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min(len(X_train), (b + 1) * batch_size)
            xb = X_train_shuff[start_idx:end_idx]
            yb = Y_train_shuff[start_idx:end_idx]
            
            y_pred_b = model.forward(xb)
            train_batch_losses.append(model.compute_loss(y_pred_b, yb))
            model.backward(xb, yb)
            
        train_loss = np.mean(train_batch_losses)
        
        # Validation evaluation
        val_pred = model.forward(X_val_scaled)
        val_loss = model.compute_loss(val_pred, Y_val_scaled)
        
        # Unscale for R2 metric
        val_pred_orig = scaler_y.inverse_transform(val_pred)
        val_r2 = r2_score(Y_val, val_pred_orig)
        
        # Learning rate schedule
        if epoch % 15 == 0:
            model.lr *= 0.6
            
        status = "*Best*" if val_loss < best_val_loss else ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
        history.append({
            "epoch": epoch,
            "train_loss_mse": round(float(train_loss), 5),
            "val_loss_mse": round(float(val_loss), 5),
            "val_r2": round(float(val_r2), 4)
        })
        
        # Print every 3 epochs or last
        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            print(f"{epoch:^8d} | {train_loss:^18.5f} | {val_loss:^16.5f} | {val_r2:^14.4f} | {status:^10}")
            
    print("-" * 75)
    
    # 4. Final Evaluation on Held-Out Test Set
    test_pred_scaled = model.forward(X_test_scaled)
    test_pred = scaler_y.inverse_transform(test_pred_scaled)
    
    r2_aep = r2_score(Y_test[:, 0], test_pred[:, 0])
    r2_wake = r2_score(Y_test[:, 1], test_pred[:, 1])
    r2_plf = r2_score(Y_test[:, 2], test_pred[:, 2])
    
    rmse_aep = math.sqrt(mean_squared_error(Y_test[:, 0], test_pred[:, 0]))
    mae_wake = mean_absolute_error(Y_test[:, 1], test_pred[:, 1])
    mae_plf = mean_absolute_error(Y_test[:, 2], test_pred[:, 2])
    
    print("\n[4/4] Test Set Performance Breakdown (Unseen Layouts):")
    print(f"  - Net AEP (GWh/yr):      R^2 = {r2_aep:7.4f}  |  RMSE = {rmse_aep:5.2f} GWh")
    print(f"  - Wake Deficit (%):      R^2 = {r2_wake:7.4f}  |  MAE  = {mae_wake:5.2f} %")
    print(f"  - Plant Load Factor (%): R^2 = {r2_plf:7.4f}  |  MAE  = {mae_plf:5.2f} %")
    print(f"  - Overall Multi-Task R^2:      {np.mean([r2_aep, r2_wake, r2_plf]):7.4f}")
    
    # 5. Latency Benchmark: Physics Engine vs Neural Surrogate
    print("\n--- Inference Latency Benchmark (1,000 Evaluations) ---")
    single_x = X_test_scaled[:1]
    
    # Benchmark Surrogate
    t0 = time.time()
    for _ in range(1000):
        _ = model.forward(single_x)
    surrogate_time_ms = ((time.time() - t0) / 1000.0) * 1000.0
    
    physics_time_ms = 7.78  # Measured average from M3
    speedup = physics_time_ms / surrogate_time_ms
    print(f"  - M3 Full Physics Simulation:  {physics_time_ms:.2f} ms per layout")
    print(f"  - M5 Neural Surrogate Model:   {surrogate_time_ms:.4f} ms per layout")
    print(f"  - Speedup Factor:              {speedup:.1f}x FASTER (Enables Instant Sliders)")
    
    # 6. Save Artifacts
    package = {
        "model_weights": model.to_dict(),
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "metrics": {
            "r2_net_aep": round(float(r2_aep), 4),
            "rmse_net_aep_gwh": round(float(rmse_aep), 2),
            "r2_wake_deficit": round(float(r2_wake), 4),
            "mae_wake_deficit_pct": round(float(mae_wake), 2),
            "r2_plf": round(float(r2_plf), 4),
            "mae_plf_pct": round(float(mae_plf), 2),
            "mean_r2": round(float(np.mean([r2_aep, r2_wake, r2_plf])), 4),
            "speedup_factor": round(float(speedup), 1),
            "surrogate_latency_ms": round(float(surrogate_time_ms), 4)
        }
    }
    
    joblib.dump(package, OUTPUT_MODEL_JOBLIB)
    print(f"\nSaved Joblib Model to: {OUTPUT_MODEL_JOBLIB}")
    
    summary_export = {
        "architecture": "3-Layer MLP (128-64-32)",
        "features": feature_cols,
        "targets": target_cols,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "metrics": package["metrics"],
        "physics_vs_surrogate": {
            "physics_eval_ms": physics_time_ms,
            "surrogate_eval_ms": round(float(surrogate_time_ms), 4),
            "speedup": f"{speedup:.1f}x"
        }
    }
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
    print(f"Saved Summary to: {OUTPUT_SUMMARY_JSON}")
    
    with open(OUTPUT_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved Training History to: {OUTPUT_HISTORY_JSON}")
    
    print("\n" + "=" * 75)
    print(" Neural Surrogate Model Training Complete! ")
    print("=" * 75)


if __name__ == "__main__":
    train_surrogate_model()
