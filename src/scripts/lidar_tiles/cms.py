import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def cms(dataset_id="cms"):
    """
    Process the CMS dataset and save it to a geojson file.

    - Assign dataset_id
    - Create geometry from bounding box coordinates in EPSG:4326
    - Remove tiles smaller than 1MB
    - Save to geojson file

    - Assign a key to it: dataset_id/filename/filename: this is where the file is stored in r2
    """
    csv_path = Path("data/raw/cms_brazil_lidar_tile_inventory.csv")
    os.makedirs("data/lidar_tiles", exist_ok=True)
    df = pd.read_csv(csv_path)
    geometry = [
        box(row["min_lon"], row["min_lat"], row["max_lon"], row["max_lat"])
        for _, row in df.iterrows()
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf["dataset_id"] = dataset_id
    gdf["key"] = gdf["filename"].apply(lambda x: f"{dataset_id}/{x}/{x}")
    gdf = gdf[gdf["file_size_mb"] >= 1]  # remove tiles smaller than 1MB
    gdf.to_file(f"data/lidar_tiles/{dataset_id}.geojson", driver="GeoJSON")
    print(f"Number of tiles: {len(gdf)}")
    print(f"Saved to data/lidar_tiles/{dataset_id}.geojson")


if __name__ == "__main__":
    cms()
