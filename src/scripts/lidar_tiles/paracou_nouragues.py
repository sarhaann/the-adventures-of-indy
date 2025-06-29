import json
import os

import boto3
import modal
import requests
from tqdm import tqdm

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode "
        "boto3 requests tqdm 'laspy[lazrs]' pyproj shapely numpy"
    )
)
app = modal.App(name="cms_download", image=image)
secret = modal.Secret.from_name(
    "r2-secret", required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
)


def object_exists(client, prefix: str, bucket_name: str) -> bool:
    try:
        client.head_object(Bucket=bucket_name, Key=prefix)
        return True
    except Exception as e:
        return False


def download_laz_file(url: str, filename: str) -> bool:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(filename, "wb") as f,
            tqdm(
                desc=filename,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False


# 1h timeout
@app.function(secrets=[secret], timeout=3600)
def download_and_process_laz_file(item: dict):
    # Extract parameters from the JSON payload
    url = item["url"]
    key = item["key"]
    filename = item["filename"]
    r2_url = item["r2_url"]
    r2_bucket_name = item["r2_bucket_name"]

    client = boto3.client(
        "s3",
        endpoint_url=r2_url,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    try:
        if object_exists(client, key, r2_bucket_name):
            return {"success": False, "error": "File already exists", "key": key}

        print("Step 1: Downloading the laz file")
        if not download_laz_file(url, filename):
            return {"success": False, "error": "Download failed", "key": key}

        # Compute polygon boundary in EPSG:4326 (WGS84)
        try:
            import laspy  # type: ignore
            import numpy as np  # type: ignore
            from pyproj import Transformer  # type: ignore
            from shapely.geometry import Polygon  # type: ignore
            from shapely.ops import transform as shp_transform  # type: ignore

            with laspy.open(filename) as las:
                pts = las.read()
                x, y = pts.x, pts.y

            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            bbox = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
                (min_x, min_y),
            ]
            poly = Polygon(bbox)

            # EPSG:2972 -> 4326
            transformer = Transformer.from_crs("EPSG:2972", "EPSG:4326", always_xy=True)

            def _tx(x_, y_, z=None):
                return transformer.transform(x_, y_)

            poly_4326 = shp_transform(_tx, poly)
            poly_coords = list(poly_4326.exterior.coords)
        except Exception as e:
            return {
                "success": False,
                "error": f"Boundary extract failed: {e}",
                "key": key,
            }

        print("Step 2: Upload to R2")
        client.upload_file(filename, r2_bucket_name, key)

        return {
            "success": True,
            "error": None,
            "key": key,
            "polygon": poly_coords,
            "filename": filename,
            "url": url,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "key": key}


@app.local_entrypoint()
def main():
    from rich import print

    from src.config import settings

    R2_URL = settings.R2_URL
    R2_BUCKET_NAME = settings.R2_BUCKET_NAME

    def run_for_dataset(dataset_name: str):
        with open(f"data/lidar_tiles/{dataset_name}_metadata.json", "r") as f:
            metadata = json.load(f)["items"]
        to_process = []
        for tile in metadata:
            filename = tile["name"]
            if not filename.lower().endswith((".laz", ".las")):
                continue
            url = tile["download"]
            key = f"fguiana_{dataset_name}/{filename}/{filename}"
            to_process.append(
                {
                    "url": url,
                    "key": key,
                    "filename": filename,
                    "r2_url": R2_URL,
                    "r2_bucket_name": R2_BUCKET_NAME,
                }
            )
        print(f"Total items to process: {len(to_process)}")
        failures, features = [], []
        for result in download_and_process_laz_file.map(to_process):
            if not result.get("success"):
                failures.append(
                    {"key": result.get("key"), "error": result.get("error")}
                )
                continue

            coords = result["polygon"]
            feature = {
                "type": "Feature",
                "properties": {
                    "key": result["key"],
                    "filename": result["filename"],
                    "url": result["url"],
                },
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
            features.append(feature)

        with open(f"data/lidar_tiles/{dataset_name}.geojson", "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

        with open(f"tmp/{dataset_name}_download_failures.json", "w") as f:
            json.dump(failures, f, indent=2)
        print(f"Completed {len(features)} successful tiles, {len(failures)} failures.")

    run_for_dataset("paracou")
    run_for_dataset("nouragues")
