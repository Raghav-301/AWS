import json
from collections import defaultdict

with open("data/turbines/classified_turbines.geojson", "r", encoding="utf-8") as f:
    d = json.load(f)

clusters = defaultdict(list)
for feat in d["features"]:
    cid = feat["properties"]["cluster_id"]
    clusters[cid].append(feat)

print(f"Total clusters: {len(clusters)}")
for cid, tlist in sorted(clusters.items()):
    n_old = sum(1 for t in tlist if t["properties"]["turbine_class"] == "old")
    n_mid = sum(1 for t in tlist if t["properties"]["turbine_class"] == "mid")
    n_mod = sum(1 for t in tlist if t["properties"]["turbine_class"] == "modern")
    lats = [t["geometry"]["coordinates"][1] for t in tlist]
    lons = [t["geometry"]["coordinates"][0] for t in tlist]
    print(f"Cluster {cid:2d}: {len(tlist):4d} turbines | Old: {n_old:4d} | Mid: {n_mid:3d} | Mod: {n_mod:2d} | Center: ({sum(lats)/len(lats):.4f}, {sum(lons)/len(lons):.4f}) | Span: {max(lats)-min(lats):.3f}° lat, {max(lons)-min(lons):.3f}° lon")
