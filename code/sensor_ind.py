import numpy as np
import glob
import os
import re

xi = 4
sensor_counts = [100, 200, 300]

# Match all files with any f index
data_files = sorted(glob.glob(f"../data/torus_N*_xi{xi}_f*.npz"))
print(f"Found {len(data_files)} total forcing files")

# Extract unique N values from filenames
n_values = sorted({int(re.search(r"N(\d+)", os.path.basename(f)).group(1)) for f in data_files})

for N in n_values:
    for n_sensors in sensor_counts:
        if n_sensors >= N:
            continue  # Cannot select more sensors than points

        sensor_idx = np.random.choice(N, size=n_sensors, replace=False)
        out_path = f"../data/sensor_indices_{N}_xi{xi}_sensors{n_sensors}.npy"

        np.save(out_path, sensor_idx)
        print(f"Saved: {out_path}")
