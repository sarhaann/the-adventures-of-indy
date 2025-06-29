import asyncio
import hashlib
import json
import os
from collections import Counter

import geopandas as gpd
import pandas as pd
import pyreadr
from pyproj import Transformer
from shapely.geometry import Point
from tqdm import tqdm

from src.config import settings
from src.scripts.prompts import (
    CLASSIFY_CONFIRMED_SITE_SYSTEM_PROMPT,
    CLASSIFY_CONFIRMED_SITE_TOOLS,
)
from src.utils import filter_points_in_amazon


def process_gdf(
    gdf: gpd.GeoDataFrame,
    dataset_id: str,
    type_value: str | None = None,
    uid_column: str | None = None,
    save_file: bool = True,
) -> gpd.GeoDataFrame:
    """
    Process a GeoDataFrame with standard operations.

    - Ensure it's in WGS84
    - Add grid_id, dataset_id, type
    - Add uid: either from specified column or hash of geometry coordinates
    - Save to geojson
    - Print number of features
    """

    # Ensure WGS84 CRS
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Add standard fields
    gdf["dataset_id"] = dataset_id

    if type_value is not None:
        gdf["type"] = type_value

    # Generate UID
    if uid_column and uid_column in gdf.columns:
        gdf["uid"] = gdf[uid_column].astype(str)
    else:
        # Use geometry coordinates for UID
        gdf["uid"] = gdf.geometry.apply(
            lambda geom: hashlib.sha1(f"{geom.x:.6f},{geom.y:.6f}".encode()).hexdigest()
        )

    print(f"Number of features: {len(gdf)}")
    original_count = len(gdf)
    gdf = filter_points_in_amazon(gdf)
    filtered_count = len(gdf)
    print(f"Number of features after filtering: {filtered_count}")
    print(
        f"% of features within Amazon boundary: {filtered_count / original_count * 100:.2f}%"
    )
    if save_file:
        # Save and report
        gdf.to_file(f"data/confirmed/{dataset_id}.geojson", driver="GeoJSON")
        print(f"Saved to data/confirmed/{dataset_id}.geojson")

    return gdf


def iriarte(dataset_id: str = "iriarte"):
    """
    Process the iriarte dataset.

    - Convert CSV to GeoJSON format
    - Transform coordinates from UTM 19S to WGS84
    - Add dataset_id, uid, type=mound_site
    """
    print("Processing iriarte dataset...")
    df = pd.read_csv("data/raw/iriarte.csv")

    transformer = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    geometries = []
    valid_rows = []
    for idx, row in df.iterrows():
        # Skip rows with missing coordinates
        if pd.isna(row["x (UTM 19S)"]) or pd.isna(row["y (UTM 19S)"]):
            continue

        # Transform coordinates from UTM to WGS84
        lon, lat = transformer.transform(row["x (UTM 19S)"], row["y (UTM 19S)"])
        # Create point geometry
        geometries.append(Point(lon, lat))
        valid_rows.append(row)

    # Create GeoDataFrame
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    gdf = gpd.GeoDataFrame(valid_df, geometry=geometries, crs="EPSG:4326")
    process_gdf(gdf, dataset_id, "mound_site", "Site Name")

    # Clean up unnecessary columns after processing
    gdf = gdf.drop(columns=["x (UTM 19S)", "y (UTM 19S)"], errors="ignore")


def coomes(dataset_id: str = "coomes"):
    """
    Process the coomes dataset.

    - Convert CSV to GeoJSON format
    - Coordinates are already in WGS84 (Latitude/Longitude)
    - Add dataset_id, uid, type=other
    """
    print("Processing coomes dataset...")
    df = pd.read_csv("data/raw/coomes.csv")

    # Convert to GeoDataFrame
    geometries = []
    valid_rows = []

    for idx, row in df.iterrows():
        # Skip rows with missing coordinates
        if pd.isna(row["Latitude"]) or pd.isna(row["Longitude"]):
            continue

        # Create point geometry (coordinates are already in WGS84)
        geometries.append(Point(row["Longitude"], row["Latitude"]))
        valid_rows.append(row)

    # Create GeoDataFrame
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    gdf = gpd.GeoDataFrame(valid_df, geometry=geometries, crs="EPSG:4326")
    process_gdf(gdf, dataset_id, "other", "Site Name")


def james_q_jacobs(dataset_id: str = "james_q_jacobs"):
    """
    Process the James Q. Jacobs dataset.

    - Coordinates are already in WGS84 (Latitude/Longitude)
    - Add dataset_id, uid
    """
    print("Processing James Q. Jacobs dataset...")
    gdf = gpd.read_file("data/confirmed/james_q_jacobs.geojson")
    # already has a type column
    gdf = process_gdf(gdf, dataset_id, None, None, save_file=False)
    # Save with _processed suffix to match original behavior
    gdf.to_file(f"data/confirmed/{dataset_id}_processed.geojson", driver="GeoJSON")
    print(f"Saved to data/confirmed/{dataset_id}_processed.geojson")


def iphan(dataset_id: str = "iphan"):
    """
    Process the IPHAN dataset.

    - Convert shapefile to GeoJSON format
    - Coordinates are already in WGS84 (Latitude/Longitude)
    - Add dataset_id, uid
    """
    print("Processing confirmed sites...")
    if not os.path.exists("data/raw/sitios/sitios.shp"):
        raise FileNotFoundError(
            "Shapefile not found at `data/raw/sitios/sitios.shp`. Please read the Appendix of the paper for more details."
        )

    gdf_iphan = gpd.read_file("data/raw/sitios/sitios.shp")
    process_gdf(gdf_iphan, dataset_id, None, "id_bem")


async def classify_confirmed_sites(output_path: str):
    """
    Classify the confirmed sites from the IPHAN database and save them to a GeoJSON file. Read Appendix.md for more details.
    """

    async def classify_site(feature, pbar):
        try:
            response = await settings.async_openai_client.responses.create(
                model="o3",
                instructions=CLASSIFY_CONFIRMED_SITE_SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": f"Site information:\n```json\n{json.dumps(feature)}\n```",
                    }
                ],
                tools=CLASSIFY_CONFIRMED_SITE_TOOLS,
                tool_choice="required",
            )
            for r in response.output:
                if r.type == "function_call":
                    classification = json.loads(r.arguments)
                    feature["properties"]["type"] = classification["category"]
                    feature["properties"]["justification"] = classification[
                        "justification"
                    ]
                    break
        except Exception as e:
            print(f"Error processing {feature['properties']['identifica']}: {e}")
        pbar.update(1)

    async def process_batch(batch, pbar):
        tasks = [classify_site(feature, pbar) for feature in batch]
        await asyncio.gather(*tasks)

    with open("data/confirmed/iphan.geojson", "r") as f:
        data = json.load(f)
        features = data["features"]

    for feature in features:
        # not really too useful for us, even after classification
        del feature["properties"]["co_iphan"]
        del feature["properties"]["no_logrado"]
        del feature["properties"]["nu_logrado"]
        del feature["properties"]["id_naturez"]
        del feature["properties"]["id_classif"]
        del feature["properties"]["id_tipo_be"]

    batch_size = 500
    with tqdm(total=len(features), desc="Classifying confirmed sites") as pbar:
        for i in range(0, len(features), batch_size):
            batch = features[i : i + batch_size]
            await process_batch(batch, pbar)

    fc = {
        "type": "FeatureCollection",
        "name": "iphan_classified",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    with open(output_path, "w") as f:
        json.dump(fc, f)
    print(f"Saved to {output_path}")


def iphan_classify():
    asyncio.run(classify_confirmed_sites("data/confirmed/iphan_classified.geojson"))
    with open("data/confirmed/iphan_classified.geojson", "r") as f:
        data = json.load(f)
        features = data["features"]
    # print summary stats by count
    print(f"Total sites: {len(features)}")
    counts = Counter(
        [f["properties"]["type"] for f in features if "type" in f["properties"]]
    )
    print(f"Counts: {counts}")
    print(f"Total sites with type: {sum(counts.values())}")
    print("Sites without type:")
    # if any features don't have a type, print them
    for f in features:
        if "type" not in f["properties"]:
            print(f["properties"])


def prumers(dataset_id: str = "prumers"):
    """
    Process the Prumers dataset.

    - Convert CSV to GeoJSON format
    - Add dataset_id, uid
    - type: `lidar_located_earthwork`
    """
    print("Processing Prumers dataset...")
    csv_path = "data/raw/prumers.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at `{csv_path}`. Please place the file there or adjust the path."
        )

    df = pd.read_csv(csv_path)
    transformer = Transformer.from_crs("EPSG:32720", "EPSG:4326", always_xy=True)

    def to_lon_lat(row):
        lon, lat = transformer.transform(row["UTM_X"], row["UTM_Y"])
        return Point(lon, lat)

    df["geometry"] = df.apply(to_lon_lat, axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    process_gdf(gdf, dataset_id, "lidar_located_earthwork", "Name")


def souza(dataset_id: str = "souza"):
    """
    Process the Souza dataset.

    - Convert CSV to GeoJSON format
    - Coordinates are already in WGS84 (Latitude/Longitude)
    - Add dataset_id, uid
    - type: `lidar_located_earthwork`
    """
    print("Processing Pre-Columbian Earth Builders dataset...")
    csv_path = "data/raw/souza.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at `{csv_path}`. Please place the file there or adjust the path."
        )

    df = pd.read_csv(csv_path)
    df["geometry"] = df.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    process_gdf(gdf, dataset_id, "lidar_located_earthwork", "Site")


def walker(dataset_id: str = "walker"):
    """
    Process the Walker dataset.

    - Convert CSV to GeoJSON format
    - Coordinates are already in WGS84 (Latitude/Longitude)
    - Add dataset_id, uid
    - uid: hash of long and lat
    - type: `earthwork` or `ade` (there is also type other but we discard it!)
    """
    print("Processing Walker dataset...")
    csv_path = "data/raw/walker.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at `{csv_path}`. Please place the file there or adjust the path."
        )

    df = pd.read_csv(csv_path)
    df = df[df["type"].isin(["earthwork", "ade"])]
    print(f"Total sites after filtering 'other' type: {len(df)}")
    df["geometry"] = df.apply(lambda row: Point(row["x"], row["y"]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    # remove all other properties
    gdf = gdf.drop(columns=gdf.columns.difference(["geometry", "type"]))
    # Walker already has type column, use coordinate hash for UID
    process_gdf(gdf, dataset_id)


def carolina(dataset_id: str = "carolina"):
    """
    Process the Carolina dataset.

    - Convert RDS to GeoJSON format
    - Coordinates are already in WGS84 (Latitude/Longitude)
    - Add dataset_id, uid
    - type: `earthwork`
    """
    print("Processing Carolina dataset...")
    rds_path = "data/raw/carolina/Earthworks.rds"
    if not os.path.exists(rds_path):
        raise FileNotFoundError(
            f"RDS not found at `{rds_path}`. Please place the file there or adjust the path."
        )

    # Read RDS file using pyreadr
    result = pyreadr.read_r(rds_path)
    df = result[None]
    df["geometry"] = df.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    # Use coordinate hash for UID (no uid_column specified)
    process_gdf(gdf, dataset_id, "earthwork")


def radambrasil(dataset_id: str = "radambrasil"):
    """
    Process the control sites from the RADAMBRASIL Latossolo Control Dataset.

    - Convert CSV to GeoJSON format
    - Coordinates are already in WGS84 (Latitude/Longitude)
    - Add dataset_id, uid
    - type: `terra_preta_control`
    """
    print(
        "Processing terra preta control sites from RADAMBRASIL Latossolo Control Dataset..."
    )

    if not os.path.exists("data/raw/radambrasil.csv"):
        raise FileNotFoundError(
            "CSV file not found at `data/raw/radambrasil.csv`. Please read the Appendix of the paper for more details."
        )

    df = pd.read_csv("data/raw/radambrasil.csv", encoding="latin1")
    df = df.dropna(axis=1, how="all")
    # Filter for Latossolo soils with valid coordinates
    latossolo = df[
        (df["Classification"].str.contains("Latossolo", case=False, na=False))
        & (df["Latitude"] > -90)
        & (df["Latitude"] < 90)
        & (df["Longitude"] > -180)
        & (df["Longitude"] < 180)
        & (df["Latitude"] != -999.999)
        & (df["Longitude"] != -999.999)
    ][["Latitude", "Longitude", "Classification"]].drop_duplicates()
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(latossolo["Longitude"], latossolo["Latitude"])
    ]
    gdf = gpd.GeoDataFrame(latossolo, geometry=geometry, crs="EPSG:4326")
    process_gdf(gdf, dataset_id, "terra_preta_control")


def wilian(dataset_id: str = "wilian"):
    """
    Process the Wilian dataset.

    - Already in geojson format
    - Coordinates are already in WGS84 (Latitude/Longitude)
    - Add dataset_id, uid
    - type: `terra_preta`
    """
    gdf = gpd.read_file("data/confirmed/wilian.geojson")
    gdf = process_gdf(gdf, dataset_id, "terra_preta", save_file=False)
    gdf.to_file(f"data/confirmed/{dataset_id}_processed.geojson", driver="GeoJSON")
    print(f"Saved to data/confirmed/{dataset_id}_processed.geojson")


if __name__ == "__main__":
    iriarte()
    coomes()
    james_q_jacobs()
    iphan()
    prumers()
    souza()
    walker()
    carolina()
    radambrasil()
    wilian()
    iphan_classify()
