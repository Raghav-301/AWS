import tifffile
import zstandard
import numpy as np

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

# Test endianness
for endian in ['<f4', '>f4']:
    fl = np.transpose(unpred, (0, 2, 1)).copy().view(endian).reshape((512, 512))
    v = fl[(fl > 1) & (fl < 30)]
    print(endian, 'Valid count:', len(v), 'mean:', v.mean() if len(v) else 0)
    print('Sample:', fl[300, 100:105])

# Also test order of planes: [3, 2, 1, 0] or [0, 1, 2, 3]
for rev in [False, True]:
    arr = unpred[:, ::-1, :] if rev else unpred
    for endian in ['<f4', '>f4']:
        fl = np.transpose(arr, (0, 2, 1)).copy().view(endian).reshape((512, 512))
        v = fl[(fl > 1) & (fl < 30)]
        print(f"rev={rev}, endian={endian} -> count: {len(v)}, mean: {v.mean() if len(v) else 0}")
