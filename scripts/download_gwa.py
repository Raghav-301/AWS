"""
Download Global Wind Atlas (GWA) GeoTIFFs for India (100m and 150m height).
URL format: https://globalwindatlas.info/api/gis/country/IND/wind-speed/{height}
"""
import os
import sys
import time
import requests

GWA_URLS = {
    "100m": "https://globalwindatlas.info/api/gis/country/IND/wind-speed/100",
    "150m": "https://globalwindatlas.info/api/gis/country/IND/wind-speed/150",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "gwa")

def download_file(url, target_path, label):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 100 * 1024 * 1024:
        print(f"[{label}] Already downloaded: {target_path} ({os.path.getsize(target_path)/(1024*1024):.1f} MB)")
        return True

    print(f"[{label}] Downloading from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=600)
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        start_time = time.time()
        last_log = start_time

        temp_path = target_path + ".tmp"
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024): # 1MB chunk
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log > 3:
                        speed = downloaded / (1024 * 1024 * (now - start_time + 0.001))
                        pct = (downloaded / total_size * 100) if total_size else 0
                        print(f"[{label}] Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s")
                        last_log = now

        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)
        print(f"[{label}] Successfully downloaded: {target_path} ({os.path.getsize(target_path)/(1024*1024):.1f} MB)")
        return True
    except Exception as e:
        print(f"[{label}] Download failed: {e}")
        return False

def main():
    print("=" * 60)
    print("GWA GeoTIFF Downloader for India (100m & 150m)")
    print("=" * 60)
    for height_label, url in GWA_URLS.items():
        out_file = os.path.join(OUT_DIR, f"ind_wind-speed_{height_label}.tif")
        success = download_file(url, out_file, height_label)
        if not success:
            print(f"Warning: Failed to download {height_label}")

if __name__ == "__main__":
    main()
