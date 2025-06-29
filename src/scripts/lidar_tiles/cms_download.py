import json
import os

import boto3
import modal
import requests
from tqdm import tqdm

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands("uv pip install --system --compile-bytecode boto3 requests tqdm")
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
            return -1, "File already exists", key

        print("Step 1: Downloading the laz file")
        if not download_laz_file(url, filename):
            return 0, "Failed to download the laz file", key
        print("Step 2: Upload to R2")
        client.upload_file(filename, r2_bucket_name, key)
        return 1, "Processing complete!", key
    except Exception as e:
        return 0, f"Failed to process the laz file: {e}", key


@app.local_entrypoint()
def main():
    from rich import print

    from src.config import settings

    R2_URL = settings.R2_URL
    R2_BUCKET_NAME = settings.R2_BUCKET_NAME

    BASE_URL = "https://daac.ornl.gov/orders/052eeb56a92c5b59670fa39e306774a5/LiDAR_Forest_Inventory_Brazil/data/"

    with open("data/lidar_tiles/cms.geojson", "r") as f:
        tiles = json.load(f)["features"]

    to_process = []
    for tile in tiles:
        filename = tile["properties"]["filename"]
        key = tile["properties"]["key"]
        url = BASE_URL + filename
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
    failures = []
    already_processed = []
    for code, message, key in download_and_process_laz_file.map(to_process):
        if code == 0:
            failures.append(f"{key}: {message}")
        elif code == -1:
            already_processed.append(key)

    with open("tmp/cms_download_failures.json", "w") as f:
        json.dump(failures, f)
    with open("tmp/cms_download_already_processed.json", "w") as f:
        json.dump(already_processed, f)
    print(f"Already processed: {len(already_processed)}")
    print(f"Failures: {failures}")
