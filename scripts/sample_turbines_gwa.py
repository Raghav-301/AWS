import json
import numpy as np
import tifffile
import zstandard

tif = tifffile.TiffFile('data/gwa/IND_wind-speed_100m.tif')
page = tif.pages[0]
offset = page.dataoffsets[484]
bytecount = page.databytecounts[484]

with open('data/gwa/IND_wind-speed_100m.tif', 'rb') as f:
    f.seek(offset)
    comp = f.read(bytecount)

raw = zstandard.ZstdDecompressor().decompress(comp, max_output_size=512*512*4)
raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape((512, 4, 512))
unpred = np.cumsum(raw_arr, axis=2, dtype=np.uint8)
tile = np.transpose(unpred, (0, 2, 1)).copy().view('>f4').reshape((512, 512))

origin_lon = 65.61875
origin_lat = 33.27875
pixel_size = 0.0025

with open("data/osm/turbines.geojson", "r", encoding="utf-8") as f:
    turbines_data = json.load(f)

turbine_speeds = []
for feat in turbines_data["features"][:200]:
    lon, lat = feat["geometry"]["coordinates"]
    r = int(round((origin_lat - lat) / pixel_size)) - 19 * 512
    c = int(round((lon - origin_lon) / pixel_size)) - 9 * 512
    if 0 <= r < 512 and 0 <= c < 512:
        val = float(tile[r, c])
        turbine_speeds.append(val)

print(f"Sampled {len(turbine_speeds)} turbines from GWA 100m raster:")
print(f"Min speed at turbines:    {min(turbine_speeds):.2f} m/s")
print(f"Max speed at turbines:    {max(turbine_speeds):.2f} m/s")
print(f"Mean speed at turbines:   {np.mean(turbine_speeds):.2f} m/s")
print(f"Median speed at turbines: {np.median(turbine_speeds):.2f} m/s")
print("First 15 sample speeds at actual turbines:", [round(x, 2) for x in turbine_speeds[:15]])
