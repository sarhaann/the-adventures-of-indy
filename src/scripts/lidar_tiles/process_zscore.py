import json
from pathlib import Path

import matplotlib.pyplot as plt
import modal
import numpy as np
import pdal
import rasterio

image = (
    modal.Image.from_registry("pdal/pdal:latest", add_python="3.12")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode boto3 numpy rasterio matplotlib opencv-python-headless requests tqdm pandas pyproj"
    )
    .run_commands("uv pip install --system --compile-bytecode pdal python-pdal gdal")
)
app = modal.App(name="lidar_tiles_process_zscore", image=image)
secret = modal.Secret.from_name(
    "r2-secret", required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
)
RESOLUTION = 1.0
R2_URL = "https://3fbfd89552e9f9346d736c3f1735bd4e.r2.cloudflarestorage.com"
R2_BUCKET_NAME = "city-of-z"


def create_zscore_anomaly_map_sparse(
    dem: np.ndarray, window_size: int = 51, min_valid_ratio: float = 0.1
) -> np.ndarray:
    """Create z-score anomaly map that handles sparse data properly"""
    # Create mask for valid pixels
    valid_mask = (dem > 0) & ~np.isnan(dem)

    # If not enough valid data overall, return NaNs
    if valid_mask.sum() < 1000:
        print("Warning: Too few valid pixels for anomaly detection")
        return np.full_like(dem, np.nan, dtype=np.float32)

    # Create output array
    result = np.full_like(dem, np.nan, dtype=np.float32)

    # Process only valid pixels
    valid_rows, valid_cols = np.where(valid_mask)

    print(f"Processing {len(valid_rows):,} valid pixels...")

    # For each valid pixel, calculate local z-score
    for i, (row, col) in enumerate(zip(valid_rows, valid_cols)):
        if i % 50000 == 0:  # Progress indicator
            print(f"  Processed {i:,}/{len(valid_rows):,} pixels...")

        # Define window bounds
        half_window = window_size // 2
        r_start = max(0, row - half_window)
        r_end = min(dem.shape[0], row + half_window + 1)
        c_start = max(0, col - half_window)
        c_end = min(dem.shape[1], col + half_window + 1)

        # Extract window
        window = dem[r_start:r_end, c_start:c_end]
        valid_in_window = window[window > 0]

        # Check if enough valid data in window
        min_valid_count = int(window_size * window_size * min_valid_ratio)
        if len(valid_in_window) < min_valid_count:
            continue

        # Calculate local statistics
        local_mean = valid_in_window.mean()
        local_std = valid_in_window.std()

        # Skip if std is too small (uniform area)
        if local_std < 1e-6:
            continue

        # Calculate z-score
        pixel_value = dem[row, col]
        z_score = (pixel_value - local_mean) / local_std
        result[row, col] = np.clip(z_score, -4, 4)

    return result


def create_smoothed_dem(ground_file: str):
    """Read ground‑only LAS, rasterise to DEM with a gentle smoothing."""
    smoothing_radius = max(3.0, RESOLUTION * 2)

    pipeline = [
        {"type": "readers.las", "filename": str(ground_file)},
        {
            "type": "writers.gdal",
            "filename": "temp_ground_dem.tif",
            "resolution": RESOLUTION,
            "output_type": "mean",
            "window_size": int(smoothing_radius * 2),
            "nodata": -9999,
        },
    ]

    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()

    with rasterio.open("temp_ground_dem.tif") as src:
        dem = src.read(1).astype(np.float32)
        dem[dem == -9999] = np.nan
        transform = src.transform
        bounds = src.bounds
        crs = src.crs

    Path("temp_ground_dem.tif").unlink(missing_ok=True)
    return dem, transform, bounds, crs


def save_with_colormap(data: np.ndarray, output_path: str, colormap: str = "viridis"):
    valid = ~np.isnan(data)
    if not np.any(valid):
        print(f"Warning: no valid data for {output_path}")
        return
    vmin, vmax = np.percentile(data[valid], [2, 98])
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    norm[~valid] = 0
    plt.figure(figsize=(10, 10), dpi=150)
    plt.imshow(norm, cmap=colormap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


# 1h timeout
@app.function(
    secrets=[secret],
    timeout=3600,
    volumes={
        "/r2": modal.CloudBucketMount(
            bucket_name=R2_BUCKET_NAME,
            bucket_endpoint_url=R2_URL,
            secret=secret,
        )
    },
)
def process_laz_file(key: str):
    parent_dir = str(Path(key).parent)  # e.g. "cms/ANA_A01_2017_laz_0.laz"

    try:
        if Path(f"/r2/{parent_dir}/zscore.jpg").exists():
            return -1, "File already exists", key

        print("Step 1: Processing the laz file")
        ground_file = "ground.laz"
        pipeline = [
            {"type": "readers.las", "filename": f"/r2/{key}"},
            {
                "type": "filters.pmf",
                "max_window_size": 33,
                "slope": 1.0,
                "max_distance": 2.5,
                "initial_distance": 0.15,
                "cell_size": 1.0,
            },
            {"type": "filters.range", "limits": "Classification[2:2]"},
            {
                "type": "filters.outlier",
                "method": "statistical",
                "multiplier": 2.0,
                "mean_k": 8,
            },
            {"type": "writers.las", "filename": str(ground_file)},
        ]
        pdal.Pipeline(
            json.dumps(
                {
                    "pipeline": pipeline,
                    "threshold": 0,
                }
            )
        ).execute()
        dem, transform, bounds, crs = create_smoothed_dem(str(ground_file))
        anomaly_map = create_zscore_anomaly_map_sparse(dem)
        save_with_colormap(anomaly_map, "zscore.jpg", "RdBu_r")
        save_with_colormap(dem, "dem.jpg", "terrain")

        # Ensure the destination "directory" exists in the FUSE mount (no-op if it already does)
        (Path("/r2") / parent_dir).mkdir(parents=True, exist_ok=True)

        # Upload the generated images – writing to the CloudBucketMount path creates/overwrites the object
        Path(f"/r2/{parent_dir}/zscore.jpg").write_bytes(
            Path("zscore.jpg").read_bytes()
        )
        Path(f"/r2/{parent_dir}/dem.jpg").write_bytes(Path("dem.jpg").read_bytes())
        return 1, "Processing complete!", key
    except Exception as e:
        return 0, f"Failed to process the laz file: {e}", key


@app.local_entrypoint()
def main():
    from rich import print

    from src.evals.lidar.data import get_missing_zscore_keys

    keys = get_missing_zscore_keys()
    print(f"Total items to process: {len(keys)}")
    # Ensure tmp directory exists for the report files
    Path("tmp").mkdir(exist_ok=True)
    failures = []
    already_processed = []
    for code, message, key in process_laz_file.map(keys):
        if code == 0:
            failures.append(f"{key}: {message}")
        elif code == -1:
            already_processed.append(key)

    with open("tmp/lidar_tiles_process_zscore_failures.json", "w") as f:
        json.dump(failures, f)
    with open("tmp/lidar_tiles_process_zscore_already_processed.json", "w") as f:
        json.dump(already_processed, f)
    print(f"Already processed: {len(already_processed)}")
    print(f"Failures: {failures}")
