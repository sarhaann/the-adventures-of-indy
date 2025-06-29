import base64
import io
import json

import geopandas as gpd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union


def encode_image_str(img_path: str) -> str:
    """Encode an image path as base64."""
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def encode_image_pil(img: Image.Image) -> str:
    """Encode a PILimage as base64."""
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        return base64.b64encode(output.getvalue()).decode("utf-8")


def get_amazon_boundary() -> Polygon:
    """Get the Amazon boundary as a shapely polygon."""
    with open("data/amazon_boundary.geojson", "r") as f:
        amazon_boundary = json.load(f)
    return unary_union(
        [shape(feat["geometry"]) for feat in amazon_boundary["features"]]
    )


def filter_points_in_amazon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filter a GeoDataFrame to only include points within the Amazon boundary.

    Args:
        gdf: GeoDataFrame with point geometries

    Returns:
        Filtered GeoDataFrame containing only points within Amazon boundary
    """
    amazon_boundary = get_amazon_boundary()
    within_amazon = gdf.geometry.within(amazon_boundary)
    return gdf[within_amazon].copy()


def haversine_np(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Vectorised haversine (great-circle) distance.

    Parameters
    ----------
    lon1, lat1 : float or np.ndarray
        Longitude/latitude of the first point(s) in decimal degrees.
    lon2, lat2 : float or np.ndarray
        Longitude/latitude of the second point(s) in decimal degrees.

    Returns
    -------
    np.ndarray
        Great-circle distance(s) in **kilometres**.
    """
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371.0  # Earth radius in kilometres
    return R * c
