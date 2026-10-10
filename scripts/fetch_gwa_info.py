"""
Global Wind Atlas — Download Instructions
==========================================
The GWA site has NO download API. Downloads must be done manually.
This script prints exactly what to download and where to save it.

Website: https://globalwindatlas.info/en/area/India
"""

import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MUPPANDAL_BBOX, GWA_DIR


def main():
    bbox = MUPPANDAL_BBOX

    print("=" * 70)
    print("  GLOBAL WIND ATLAS — Manual Download Instructions")
    print("=" * 70)
    print()
    print("  The GWA site does NOT have a download API.")
    print("  You must download GeoTIFF rasters manually.")
    print()
    print("  STEP 1: Go to https://globalwindatlas.info/en/area/India")
    print()
    print("  STEP 2: Navigate to the Muppandal area:")
    print(f"          Center: ({bbox['center_lat']}, {bbox['center_lon']})")
    print(f"          Bbox:   S={bbox['south']}, N={bbox['north']}, "
          f"W={bbox['west']}, E={bbox['east']}")
    print()
    print("  STEP 3: Download these layers as GeoTIFF (click the area -> Download):")
    print()
    print("  ┌─────┬────────────────────────────────┬────────────────────────────────┐")
    print("  │  #  │  Layer                         │  Save As                       │")
    print("  ├─────┼────────────────────────────────┼────────────────────────────────┤")
    print("  │  1  │  Wind speed at 100 m           │  gwa_wind_speed_100m.tif       │")
    print("  │  2  │  Wind speed at 150 m           │  gwa_wind_speed_150m.tif       │")
    print("  │  3  │  Power density at 100 m        │  gwa_power_density_100m.tif    │")
    print("  │  4  │  Elevation                     │  gwa_elevation.tif             │")
    print("  │  5  │  Ruggedness index               │  gwa_ruggedness.tif            │")
    print("  │  6  │  Air density                   │  gwa_air_density.tif           │")
    print("  └─────┴────────────────────────────────┴────────────────────────────────┘")
    print()
    print(f"  STEP 4: Save all files to:")
    print(f"          {GWA_DIR}")
    print()
    print("  NOTES:")
    print("  - Select 'India' as the country first")
    print("  - You can download for the whole country or a specific region")
    print("  - Files are CC-BY 4.0 licensed (cite: DTU + World Bank)")
    print("  - Resolution: 250m grid cells")
    print("  - If the site is slow, wind speed at 100m + 150m are the minimum needed")
    print()

    # Check what's already downloaded
    if os.path.exists(GWA_DIR):
        existing = os.listdir(GWA_DIR)
        if existing:
            print("  Currently in GWA directory:")
            for f in existing:
                size = os.path.getsize(os.path.join(GWA_DIR, f))
                print(f"    ✅ {f} ({size / 1024:.1f} KB)")
        else:
            print("  ❌ GWA directory is empty — downloads needed!")
    else:
        print("  ❌ GWA directory doesn't exist yet")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
