# PawanSthal (पवनस्थल)
### State Wind Repowering & Spacing Decision Support Engine
**Track: Waste and Energy | Target: Ministry of New & Renewable Energy (MNRE) Repowering Policy 2023**

---

## 1. Executive Summary & Policy Background

India has **25.406 GW** of wind repowering potential (National Institute of Wind Energy — NIWE), concentrated in prime wind corridors where early turbines were installed in the 1990s and 2000s. These legacy turbines are obsolete:
- **Sub-MW capacities** (mostly 250–500 kW machines)
- **Low hub heights** (30–50 m towers capturing only boundary-layer wind)
- **Tightly packed micro-siting** (2–3 rotor diameters apart), creating severe aerodynamic wake shadowing that burns up to **20% of gross energy** (NREL benchmark).

In December 2023, the Ministry of New & Renewable Energy (MNRE) issued the revised **National Repowering & Life Extension Policy for Wind Power Projects – 2023**. The policy mandates State Nodal Agencies (SNAs like TEDA in Tamil Nadu, GEDA in Gujarat) and Central Nodal Agencies (NIWE):
1. **Identify underperforming sub-2 MW clusters** for repowering.
2. **Enforce the 1.5× Generation Multiplier**: A repowered wind project must produce at least $\mathbf{1.5\times}$ the historical 3-year baseline Annual Energy Production (AEP).
3. **Value Buyouts via RSPC**: Calculate Repowering Site Preparation Cost (RSPC) including terminal valuation, PPA termination incentives, and scrap recovery.

**PawanSthal** is an automated decision-support engine that combines spatial OpenStreetMap infrastructure, NASA POWER climate reanalysis, physics-based Jensen wake equations, and deep learning surrogate models to rank repowering packages, optimize modern 3 MW+ layouts, and run real-time what-if scenario simulations for state officials.

---

## 2. Machine Learning & Modeling Architecture

PawanSthal implements an end-to-end pipeline spanning **5 interlinked modules**:

```
[OpenStreetMap Turbines & GIS] ---> [M1: Random Forest Classifier] ---> Identified 1,629 Eligible Turbines (99.6%)
                                                                           |
[NASA POWER + GWA Data]       ---> [M2: Wind Downscaling Regressor]   ---> High-Res Hub Wind Field (R² = 0.974)
                                                                           |
[Wake Aerodynamics Physics]    ---> [M3: Vectorized Jensen Engine]    ---> Hourly Directional Wake Deficits
                                                                           |
[Spatial K-Means & Setbacks]   ---> [M4: Layout Optimizer]            ---> Setback-Compliant 3MW Layouts
                                                                           |
[1,800 Physics Simulations]    ---> [M5: Deep Neural Surrogate]       ---> 406x Faster AEP & Wake Inferences
```

---

## 3. Detailed Model Specifications

### Module M1: Turbine Classification & Age Estimation
* **File**: [`scripts/train_m1_classifier.py`](scripts/train_m1_classifier.py)
* **Model Type**: `RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)`
* **Task**: Classify mapped OpenStreetMap turbines into three operational classes:
  - `old`: $<1\text{ MW}$ (obsolete 250–500 kW machines, eligible for repowering)
  - `mid`: $1\text{--}2\text{ MW}$ (transitional machines, eligible under 2023 policy)
  - `modern`: $>2\text{ MW}$ (modern 2.5–3.5 MW machines, already optimized)

#### Input Features (10 Spatial Proximity Features)
| Feature Name | Description | Unit |
| :--- | :--- | :--- |
| `nn1_dist_m` | Distance to 1st nearest neighbor turbine | Meters |
| `nn2_dist_m` | Distance to 2nd nearest neighbor turbine | Meters |
| `nn3_dist_m` | Distance to 3rd nearest neighbor turbine | Meters |
| `nn5_dist_m` | Distance to 5th nearest neighbor turbine | Meters |
| `mean_nn3_m` | Mean distance of the 3 closest neighboring turbines | Meters |
| `density_300m` | Number of neighboring turbines within 300 m radius | Count |
| `density_500m` | Number of neighboring turbines within 500 m radius | Count |
| `density_1000m`| Number of neighboring turbines within 1,000 m radius | Count |
| `dist_substation_m` | Proximity to nearest electrical substation | Meters |
| `dist_road_m` | Proximity to nearest public road / access way | Meters |

#### Training Configuration & Loss Metrics
- **Dataset**: 1,636 turbines from OpenStreetMap in the Muppandal Wind Corridor.
- **Split**: 80% Train (1,308 samples) / 20% Validation (328 samples).
- **Train Cross-Entropy Loss**: `0.0033` | **Train Accuracy**: `100.00%`
- **Validation Cross-Entropy Loss**: `0.0106` | **Validation Accuracy**: `99.70%`
- **Out-of-Bag (OOB) Score**: `99.77%`

#### Feature Importance Ranking
1. **`nn2_dist_m`**: `28.11%` (Second-neighbor spacing strongly separates staggered modern grids from packed legacy lines)
2. **`density_300m`**: `26.97%` (Extreme density $\ge 2$ within 300 m is a definitive indicator of pre-2000 sub-MW machines)
3. **`mean_nn3_m`**: `17.80%`
4. **`nn1_dist_m`**: `12.82%`
5. **`nn3_dist_m`**: `9.16%`
6. `density_500m`, `nn5_dist_m`, `dist_substation_m`, `dist_road_m`: $< 5\%$ cumulative

---

### Module M2: High-Resolution Wind Resource Downscaling
* **File**: [`scripts/m2_wind_downscaling.py`](scripts/m2_wind_downscaling.py)
* **Model Type**: `HistGradientBoostingRegressor(max_iter=150, lr=0.08, max_depth=6)`
* **Task**: Downscales coarse regional NASA POWER satellite reanalysis (0.5° × 0.5° grid) to fine-grained 100 m, 120 m, and 150 m hub-height wind speed fields across the Aralvaimozhi Pass mountain gap.

#### Input Features (6 Geospatial Features)
1. `lat`: Latitude
2. `lon`: Longitude
3. `dist_pass_km`: Euclidean distance to the Aralvaimozhi Venturi Pass gap center ($8.22^\circ\text{ N}, 77.53^\circ\text{ E}$)
4. `coarse_ws50m`: Interpolated coarse NASA POWER wind speed at 50 m (m/s)
5. `coarse_ws2m`: Surface wind speed at 2 m (m/s)
6. `coarse_t2m`: 2 m air temperature (°C)

#### Target Outputs
- High-resolution wind speed at $100\text{ m}$, $120\text{ m}$, and $150\text{ m}$ hub heights, calibrated against Global Wind Atlas (GWA 3.0) 250 m micro-scale rasters.

#### Evaluation Metrics
- **Coefficient of Determination ($R^2$)**: $\mathbf{0.974}$
- **Root Mean Squared Error (RMSE)**: $\mathbf{0.082\text{ m/s}}$
- **Mean Absolute Error (MAE)**: $\mathbf{0.061\text{ m/s}}$

---

### Module M3: Aerodynamic Wake Physics Engine
* **File**: [`scripts/m3_wake_model.py`](scripts/m3_wake_model.py)
* **Algorithm**: Vectorized Jensen / NREL Park Kinematic Wake Deficit Engine.
* **Physics Formulation**:
  $$\frac{\Delta v}{v_\infty} = \left( 1 - \sqrt{1 - C_t} \right) \left( \frac{r_0}{r_0 + k_{\text{decay}} \cdot \Delta x_{\text{downwind}}} \right)^2 \cdot \frac{A_{\text{overlap}}}{A_{\text{rotor}}}$$
  $$\Delta v_{\text{total}} = \sqrt{\sum_i \Delta v_i^2} \quad \text{(Root-Sum-Square velocity deficit combination)}$$
- **Power Curve**: Piecewise cubic generator curve parameterized with cut-in ($3.0\text{ m/s}$), rated ($10.5\text{ m/s}$), and cut-out ($25.0\text{ m/s}$) speeds.
- **Wind Rose Integration**: Full 16-sector meteorological directional integration across 8,760 annual hours using 2023 hourly NASA POWER observations.

---

### Module M4: Cluster Siting & Policy Decision Engine
* **File**: [`scripts/m4_optimizer.py`](scripts/m4_optimizer.py)
* **Optimization & Constraints**:
  - Partitions regional turbines into 6 contiguous repowering zones via spatial K-Means.
  - Implements staggered micro-siting spacing: **$5.5D$ downwind** along dominant Westerly axis ($270^\circ$) and **$3.5D$ crosswind**.
  - **Setback Enforcements**: Minimum $200\text{ m}$ exclusion buffer from OSM roads and $500\text{ m}$ buffer from settlements.
  - **Policy Validation**: Tests each zone against the MNRE 2023 $1.5\times$ threshold:
    $$\text{Multiplier} = \frac{\text{Net AEP}_{\text{repowered}}}{\text{Net AEP}_{\text{baseline}}} \ge 1.50$$
  - **Financial Metrics**:
    - **Wind CAPEX**: ₹6.0 Crore / MW (CTUIL benchmark, Sep 2026)
    - **RSPC Buyout**: ₹0.85 Crore / MW of decommissioned capacity
    - **Grid CO₂ Offset**: $0.716\text{ kg CO}_2/\text{kWh}$ (CEA Baseline Database v19)
    - **Homes Powered**: $1,200\text{ kWh/year}$ average connection (BEE / CLASP benchmark)
    - **Tariff Model**: Old PPA tariff ₹4.16/kWh vs. Repowered Tariff ₹3.00/kWh (TNERC)

---

### Module M5: Deep Neural Surrogate Wake & AEP Engine
* **File**: [`scripts/train_surrogate.py`](scripts/train_surrogate.py)
* **Architecture**: **3-Layer Deep Multi-Layer Perceptron (MLP)**
  $$\text{Input}(7) \longrightarrow \text{Dense}(128, \text{ReLU}) \longrightarrow \text{Dense}(64, \text{ReLU}) \longrightarrow \text{Dense}(32, \text{ReLU}) \longrightarrow \text{Output}(3)$$
* **Optimizer**: Mini-batch Adam ($\text{lr} = 0.005$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, step-decay at epoch 15 & 30).
* **Why it was built**: Physics wake calculations across 16 wind sectors require $\sim 8\text{ ms}$ per layout, creating noticeable UI stutter when users drag live what-if sliders. The neural surrogate approximates wake interactions with $>98\%$ fidelity in **$0.019\text{ ms}$** ($\mathbf{406.7\times\text{ faster}}$).

#### Input Features (7 Parameters)
1. `n_turbines`: Number of repowered turbines ($12 \le N \le 45$)
2. `turbine_capacity_mw`: Nameplate rating ($1.8 \le P \le 3.6\text{ MW}$)
3. `hub_height_m`: Hub height ($90 \le H \le 150\text{ m}$)
4. `rotor_diameter_m`: Rotor diameter ($93 \le D \le 131\text{ m}$)
5. `spacing_diameters`: Relative spacing factor ($3.5 \le s \le 8.5\text{ D}$)
6. `density_per_km2`: Spatial packing density ($\text{turbines/km}^2$)
7. `grid_aspect`: Layout aspect ratio

#### Multi-Target Outputs
1. `net_aep_gwh`: Annual net electrical yield accounting for aerodynamic wake shadowing
2. `wake_loss_pct`: Percentage of gross energy lost to upstream wakes
3. `plf_pct`: Plant Load Factor (%)

#### Training Loss Progression (Epoch-by-Epoch)
```
  Epoch  |  Train Loss (MSE)  |  Val Loss (MSE)  | Val R^2 (Mean) |   Status  
---------------------------------------------------------------------------
    1    |      0.39309       |     0.07547      |     0.9184     |   *Best*  
    3    |      0.02541       |     0.02142      |     0.9765     |   *Best*  
    6    |      0.01438       |     0.01523      |     0.9831     |           
    9    |      0.01193       |     0.01320      |     0.9856     |           
   12    |      0.01047       |     0.01242      |     0.9862     |   *Best*  
   15    |      0.00988       |     0.01240      |     0.9864     |   *Best*  
   18    |      0.00835       |     0.01066      |     0.9881     |   *Best*  
   33    |      0.00643       |     0.01058      |     0.9882     |   *Best*  
   45    |      0.00591       |     0.01074      |     0.9880     | Converged
```

#### Test Set Performance on Unseen Wind Farms
- **Net AEP (GWh/yr)**: $R^2 = \mathbf{0.9956}$ | $\text{RMSE} = 2.65\text{ GWh}$
- **Wake Deficit (%)**: $R^2 = \mathbf{0.9830}$ | $\text{MAE} = 0.87\%$
- **Plant Load Factor (%)**: $R^2 = \mathbf{0.9872}$ | $\text{MAE} = 0.21\%$
- **Overall Multi-Task $R^2$**: $\mathbf{0.9886}$

#### Latency Benchmark (1,000 Evaluations)
- **M3 Physics Engine**: $7.78\text{ ms}$ per layout
- **M5 Neural Surrogate**: $0.0191\text{ ms}$ per layout
- **Inference Speedup**: **$406.7\times$ FASTER**

---

## 4. Key Muppandal Repowering Findings

| Metric | Existing Baseline | PawanSthal Proposed Repowered | Net Gain |
| :--- | :--- | :--- | :--- |
| **Turbine Count** | 1,636 small turbines | 184 modern 3 MW turbines | **$-88.7\%$ fewer turbines** |
| **Installed Capacity** | 818.0 MW | 552.0 MW | Optimized footprint |
| **Average Hub Height** | 45–50 m | 120–140 m | $+70\text{ m}$ higher |
| **Annual Energy (AEP)** | ~880 GWh/year | **3,150 GWh/year** | **$+3.58\times$ Generation** |
| **MNRE Compliance** | N/A | **100% Pass** ($>1.5\times$ in all 6 zones) | Exceeds policy minimum |
| **CO₂ Avoided** | Baseline | **1,625,000 Tonnes / Year** | Offset fossil fuel generation |
| **Homes Powered** | ~730,000 | **2,625,000 Homes** | Clean power |
| **Payback Period** | N/A | **4.8 to 6.2 Years** | Rapid financial amortization |

---

## 5. Repository File Structure

```
AWS/
├── config.py                     # All policy thresholds, CEA emission factors, CAPEX, tariffs
├── server.py                     # Flask REST API server with /api/simulate & /api/summary
├── requirements.txt              # Core python dependencies
├── README.md                     # Comprehensive architecture and model documentation
├── data/
│   ├── models/                   # Serialized trained model binaries
│   │   ├── m1_turbine_rf.joblib  # Trained M1 Random Forest model
│   │   └── m5_surrogate_model.joblib # Trained M5 Deep Neural Surrogate weights & scalers
│   ├── clusters/                 # Ranked repowering zones & proposed modern layouts GeoJSON
│   ├── turbines/                 # 1,636 OSM turbines + classification metadata GeoJSON
│   ├── osm/                      # Substations, power lines, roads, settlements GeoJSON
│   ├── nasa_power/               # 2023 hourly climate reanalysis grid & wind rose summary
│   ├── m1_rf_metrics.json        # Loss, accuracy, and feature importances for M1
│   ├── m2_downscaling_summary.json# Downscaling R2 and RMSE summary
│   ├── m5_surrogate_summary.json # Neural surrogate test R2, RMSE, and latency speedup
│   └── wind_heatmap_grid.json    # High-resolution 100/150m wind speed grid points
├── scripts/
│   ├── train_m1_classifier.py    # Trains M1 Random Forest with loss & feature importances
│   ├── train_surrogate.py        # Generates 1,800 layouts & trains M5 Neural Surrogate
│   ├── m1_turbine_classifier.py  # Spatial k-NN feature extraction on OSM turbines
│   ├── m2_wind_downscaling.py    # NASA POWER to GWA wind speed downscaling regressor
│   ├── m3_wake_model.py          # Vectorized Jensen / Park aerodynamic wake physics
│   └── m4_optimizer.py           # K-Means zone clustering & setback-aware layout generator
└── static/
    ├── index.html                # Interactive Leaflet Decision Support Dashboard
    ├── css/style.css             # Glassmorphic dark theme CSS styling
    └── js/app.js                 # Map rendering, layer controls, and real-time what-if sliders
```

---

## 6. How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt scikit-learn joblib
```

### 2. Retrain Models (Optional, runs in seconds)
```bash
# Train M1 Random Forest Turbine Classifier (~8s)
python scripts/train_m1_classifier.py

# Train M5 Deep Neural Surrogate Engine (~15s)
python scripts/train_surrogate.py
```

### 3. Start the Web Server
```bash
python server.py
```
Open your browser at **`http://127.0.0.1:5000`** to access the decision support dashboard.

---

## 7. REST API Endpoints

- `GET /api/summary`: System-wide KPIs, turbine counts, total AEP gain, CO₂ avoided, and active model metrics.
- `GET /api/clusters`: Ranked list of repowering zones with financial feasibility and generation multipliers.
- `GET /api/turbines/classified`: GeoJSON of all 1,636 existing turbines styled by class (`old` = red, `mid` = yellow, `modern` = green).
- `GET /api/turbines/repowered`: GeoJSON of the proposed 3 MW optimized repowered turbine layouts.
- `GET /api/infrastructure`: GeoJSON layers for regional substations and high-voltage transmission lines.
- `GET /api/wind/heatmap`: High-resolution downscaled wind speed grid ($100\text{ m}$ hub height).
- `POST /api/simulate`: Live what-if calculation taking `{ zone_id, hub_height_m, turbine_capacity_mw, spacing_diameters, n_turbines }` and returning real-time wake deficit, net AEP, policy $1.5\times$ pass status, CAPEX, and payback years in $<1\text{ ms}$.

---

## 8. Primary Benchmark Citations
1. **MNRE**: *National Repowering & Life Extension Policy for Wind Power Projects – 2023*, Ministry of New and Renewable Energy, Govt of India.
2. **CEA**: *CO₂ Baseline Database for the Indian Power Sector*, Version 19 (Grid factor: $0.716\text{ kg CO}_2/\text{kWh}$).
3. **CTUIL**: *Central Transmission Utility of India Ltd Advisory (September 2026)* — Onshore wind benchmark CAPEX: ₹6.0 Crore / MW.
4. **TNERC**: *Tamil Nadu Electricity Regulatory Commission Comprehensive Tariff Orders* (Old tariff: ₹4.16/kWh, Repowered tariff: ₹3.00/kWh).
5. **NREL / DTU**: Jensen, N. O. (1983), *A coupled spectral and kinematic model of the wake behind a wind turbine*.
