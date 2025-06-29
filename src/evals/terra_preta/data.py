import hashlib
import json
import random

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point, Polygon, shape
from tqdm import tqdm

random.seed(41)


def _get_all_terra_preta_confirmed_sites() -> list[dict]:
    """
    Returns all terra preta confirmed sites.
    """
    features = []
    with open("data/confirmed/iphan_classified.geojson", "r") as f:
        data = json.load(f)["features"]
        # if potential_terra_preta or high_probability_terra_preta, add to features
        for feature in data:
            if feature["properties"]["type"] == "high_probability_terra_preta":
                features.append(feature)

    with open("data/confirmed/walker.geojson", "r") as f:
        # if ade then add it to features
        walker_data = json.load(f)["features"]
        for feature in walker_data:
            if feature["properties"]["type"] == "ade":
                features.append(feature)

    with open("data/confirmed/wilian_processed.geojson", "r") as f:
        wilian_data = json.load(f)["features"]
        for feature in wilian_data:
            if feature["properties"]["type"] == "terra_preta":
                features.append(feature)
    return features


def _get_all_terra_preta_control_sites() -> list[dict]:
    """
    Returns all terra preta control sites.
    TOTAL: 171
    """
    features = []
    with open("data/confirmed/radambrasil.geojson", "r") as f:
        data = json.load(f)["features"]
        features.extend(data)
    return features


def get_confirmed_and_control_sites_for_terra_preta():
    """
    1. Get the confirmed and control sites
    2. For each site create a grid around the centroid, the grid should be 1x1 km square.
    3. Create 8 neighboring 1x1 km grids around each central grid (3x3 pattern)
    4. return two lists: confirmed_sites and control_sites
       Each item has: feature (entire site obj), geometry (shapely Polygon), id (string), neighbors (dict of str: Polygon)

    imp: ignore existing grid_id, that is redundant now!
    """
    confirmed_sites = _get_all_terra_preta_confirmed_sites()
    control_sites = _get_all_terra_preta_control_sites()

    print(
        f"Initial counts - Confirmed: {len(confirmed_sites)}, Control: {len(control_sites)}"
    )

    # Create transformer for coordinate conversion (WGS84 to UTM)
    # Using UTM zone 20S which covers most of the Amazon region
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32720", always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(
        "EPSG:32720", "EPSG:4326", always_xy=True
    )

    confirmed_data = []
    control_data = []

    def create_1km_grid(centroid_lat, centroid_lon):
        """Create a 1x1 km grid around the centroid"""
        # Convert to UTM for accurate distance calculations
        utm_x, utm_y = transformer_to_utm.transform(centroid_lon, centroid_lat)

        # Create 1km x 1km square (500m in each direction from center)
        half_km = 500  # 500 meters
        corners_utm = [
            (utm_x - half_km, utm_y - half_km),  # bottom-left
            (utm_x + half_km, utm_y - half_km),  # bottom-right
            (utm_x + half_km, utm_y + half_km),  # top-right
            (utm_x - half_km, utm_y + half_km),  # top-left
            (utm_x - half_km, utm_y - half_km),  # close the polygon
        ]

        # Convert back to WGS84
        corners_wgs84 = [transformer_to_wgs84.transform(x, y) for x, y in corners_utm]

        return Polygon(corners_wgs84)

    def create_neighbor_grids(centroid_lat, centroid_lon):
        """Create 8 neighboring 1x1 km grids around the centroid"""
        # Convert to UTM for accurate distance calculations
        utm_x, utm_y = transformer_to_utm.transform(centroid_lon, centroid_lat)

        # Define the 8 neighbor offsets (1km = 1000m in each direction)
        neighbor_offsets = [
            (-1000, -1000),  # Southwest
            (0, -1000),  # South
            (1000, -1000),  # Southeast
            (-1000, 0),  # West
            (1000, 0),  # East
            (-1000, 1000),  # Northwest
            (0, 1000),  # North
            (1000, 1000),  # Northeast
        ]

        neighbor_names = [
            "southwest",
            "south",
            "southeast",
            "west",
            "east",
            "northwest",
            "north",
            "northeast",
        ]

        neighbor_grids = {}
        half_km = 500  # 500 meters

        for i, (offset_x, offset_y) in enumerate(neighbor_offsets):
            # Calculate neighbor center
            neighbor_utm_x = utm_x + offset_x
            neighbor_utm_y = utm_y + offset_y

            # Create 1km x 1km square around neighbor center
            corners_utm = [
                (neighbor_utm_x - half_km, neighbor_utm_y - half_km),  # bottom-left
                (neighbor_utm_x + half_km, neighbor_utm_y - half_km),  # bottom-right
                (neighbor_utm_x + half_km, neighbor_utm_y + half_km),  # top-right
                (neighbor_utm_x - half_km, neighbor_utm_y + half_km),  # top-left
                (
                    neighbor_utm_x - half_km,
                    neighbor_utm_y - half_km,
                ),  # close the polygon
            ]

            # Convert back to WGS84
            corners_wgs84 = [
                transformer_to_wgs84.transform(x, y) for x, y in corners_utm
            ]
            neighbor_grids[neighbor_names[i]] = Polygon(corners_wgs84)

        return neighbor_grids

    # Process confirmed sites
    for site in tqdm(confirmed_sites, desc="Processing confirmed sites"):
        # Get the geometry and calculate centroid
        geom = shape(site["geometry"])
        centroid = geom.centroid

        # Create 1km grid around centroid
        grid_geom = create_1km_grid(centroid.y, centroid.x)

        # Create 8 neighboring grids
        neighbor_grids = create_neighbor_grids(centroid.y, centroid.x)

        # Create hash of centroid coordinates for unique ID
        centroid_str = f"{centroid.x:.6f},{centroid.y:.6f}"
        site_id = hashlib.md5(centroid_str.encode()).hexdigest()[:8]

        confirmed_data.append(
            {
                "feature": site,
                "geometry": grid_geom,
                "id": site_id,
                "neighbors": neighbor_grids,
            }
        )

    # Process control sites
    for site in tqdm(control_sites, desc="Processing control sites"):
        # Get the geometry and calculate centroid
        geom = shape(site["geometry"])
        centroid = geom.centroid

        # Create 1km grid around centroid
        grid_geom = create_1km_grid(centroid.y, centroid.x)

        # Create 8 neighboring grids
        neighbor_grids = create_neighbor_grids(centroid.y, centroid.x)

        # Create hash of centroid coordinates for unique ID
        centroid_str = f"{centroid.x:.6f},{centroid.y:.6f}"
        site_id = hashlib.md5(centroid_str.encode()).hexdigest()[:8]

        control_data.append(
            {
                "feature": site,
                "geometry": grid_geom,
                "id": site_id,
                "neighbors": neighbor_grids,
            }
        )

    print(
        f"Final data contains {len(confirmed_data)} confirmed sites and {len(control_data)} control sites"
    )

    return confirmed_data, control_data
