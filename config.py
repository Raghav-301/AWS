"""
PawanSthal — Configuration & Cited Constants
=============================================
All numbers are cited; sources noted inline.
"""

# ─── Muppandal Demo Cluster (Kanyakumari district, Tamil Nadu) ───────────────
# Bounding box confirmed from Global Wind Atlas / Google Maps
# The Aralvaimozhi Pass wind corridor, one of India's densest wind zones.
MUPPANDAL_BBOX = {
    "name": "Muppandal Wind Farm, Tamil Nadu",
    "south": 8.08,   # lat min
    "north": 8.30,   # lat max  (covers the main cluster + Aralvaimozhi Pass)
    "west":  77.34,  # lon min
    "east":  77.60,  # lon max
    "center_lat": 8.19,
    "center_lon": 77.47,
}

# ─── Grid CO₂ Emission Factor ────────────────────────────────────────────────
# Source: CEA CO₂ Baseline Database for the Indian Power Sector
# Weighted average emission factor (combined margin).
# The latest CEA database reports ~0.71 kg CO₂/kWh (declining yearly as
# renewables grow). Using 0.716 from the v19 database.
# Cite: CEA, "CO₂ Baseline Database for the Indian Power Sector", Version 19
CO2_EMISSION_FACTOR_KG_PER_KWH = 0.716

# ─── Wind Energy CAPEX ───────────────────────────────────────────────────────
# Source: CTUIL benchmark (Sep 2026): ₹6.0 crore/MW for onshore wind.
# Range from industry: ₹4.5–6.5 Cr/MW; we use the CTUIL benchmark.
# Cite: CTUIL Advisory dated September 2026; CERC RE Tariff Regulations 2024
WIND_CAPEX_CR_PER_MW = 6.0  # ₹ crore per MW
WIND_CAPEX_INR_PER_MW = WIND_CAPEX_CR_PER_MW * 1e7  # ₹ per MW

# ─── Tamil Nadu Wind Tariff ──────────────────────────────────────────────────
# TNERC shifted to competitive bidding; historical generic tariff was ₹2.86/kWh
# (TNERC Order 2018). For old PPAs, the earlier tariff of ₹4.16/kWh (2016 order)
# is more relevant for existing sub-MW turbines.
# For new repowered projects, use recent SECI auction-discovered tariff ~₹2.90–3.10/kWh.
# Cite: TNERC Comprehensive Tariff Order 2018; SECI wind auction results
TARIFF_OLD_INR_PER_KWH = 4.16     # existing old PPA tariff (TNERC 2016)
TARIFF_NEW_INR_PER_KWH = 3.00     # competitive bid tariff for repowered project
TARIFF_DEFAULT_INR_PER_KWH = 3.00 # used for payback calculations

# ─── Average Household Electricity Consumption ──────────────────────────────
# Source: CLASP & BEE 2024 survey: ~1,805 kWh/year (sampled households).
# National per-connection average: ~1,200 kWh/year.
# We use 1,200 (conservative, per-connection) for "homes powered" metric.
# Cite: CLASP-BEE Household Survey 2024; Down To Earth, 2024
AVG_HOUSEHOLD_KWH_PER_YEAR = 1200

# ─── Repowering Policy Parameters (MNRE 2023 Policy) ────────────────────────
# Cite: MNRE, "National Repowering & Life Extension Policy for Wind Power
#       Projects – 2023"
REPOWER_GENERATION_MULTIPLIER = 1.5    # new project must produce ≥1.5× old
REPOWER_MAX_PAUSE_YEARS = 2            # max PPA pause during repowering
REPOWER_COMMISSIONING_MONTHS = 24      # must commission within 24 months
ELIGIBLE_CAPACITY_THRESHOLD_MW = 2.0   # turbines below this qualify
ELIGIBLE_AGE_YEARS = 15                # turbines older than this can opt in

# ─── Old Turbine Defaults (for demo cluster) ────────────────────────────────
# Source: CEEW — pre-2000 turbines at India's best sites are mostly <500 kW,
# hubs of 50 m or lower.
OLD_TURBINE_CAPACITY_KW = 500
OLD_TURBINE_HUB_HEIGHT_M = 50
OLD_TURBINE_ROTOR_DIAMETER_M = 40  # approximate for sub-MW machines

# ─── Modern Turbine Defaults (repowering target) ────────────────────────────
NEW_TURBINE_CAPACITY_KW = 3000     # 3 MW class
NEW_TURBINE_HUB_HEIGHT_M = 120
NEW_TURBINE_ROTOR_DIAMETER_M = 120

# ─── Layout Spacing Rules of Thumb ──────────────────────────────────────────
# Industry standard: 5–9D downwind, 3–5D crosswind (rotor diameters)
SPACING_DOWNWIND_MIN_D = 5
SPACING_DOWNWIND_MAX_D = 9
SPACING_CROSSWIND_MIN_D = 3
SPACING_CROSSWIND_MAX_D = 5

# ─── Cluster Fragmentation Inputs ───────────────────────────────────────────
# These lower the priority score (more owners = harder to aggregate).
# Default values for the Muppandal demo.
DEFAULT_OWNER_COUNT = 5           # estimated number of turbine owners in cluster
IS_11KV_CONNECTION = True          # most old farms connect to 11 kV lines
# True = extra grid upgrade cost, lowers priority

# ─── RSPC Valuation Defaults (from policy Annex worked example) ──────────────
# Repowering Site Preparation Cost formula parameters
RSPC_PLF_PERCENT = 12.3           # Plant Load Factor from policy example
RSPC_TARIFF_INR_PER_KWH = 3.00   # tariff used in policy example
RSPC_INCENTIVE_PERCENT = 10       # PPA termination incentive (~10%)
RSPC_DISCOUNT_RATE = 0.08         # 8% discount for NPV

# ─── NASA POWER API ──────────────────────────────────────────────────────────
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_POWER_PARAMETERS = "WS50M,WD50M,WS2M,T2M"  # wind speed, direction, temp
NASA_POWER_COMMUNITY = "RE"  # Renewable Energy community

# ─── File Paths ──────────────────────────────────────────────────────────────
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GWA_DIR = os.path.join(DATA_DIR, "gwa")
NASA_DIR = os.path.join(DATA_DIR, "nasa_power")
OSM_DIR = os.path.join(DATA_DIR, "osm")
TURBINE_DIR = os.path.join(DATA_DIR, "turbines")
