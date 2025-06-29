import hashlib
import io
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import ee
import folium
import geemap.foliumap as geemap
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import requests
from filelock import FileLock
from openai.types.responses import ResponseOutputItem
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from shapely.geometry import Polygon

from src.config import settings
from src.utils import encode_image_pil

# Use non-GUI backend so matplotlib does not try to open MacOSX windows when
# running from worker threads / headless environments.
matplotlib.use("Agg")

# We use DIMENSIONS=768x768 for all our images/maps: o3 and 4.1 process images like this (detail='high'):
# - First: scale down image to fit in 2048x2048
# - Then: scale so image's shorter side is 768px.
# - Then: calculate how many 512x512 patches can fit in the image.
# - Number of tokens = (number of patches \* per_patch_tokens) + fixed_tokens
# - So to make it easier, we set the dimensions to 768x768. We can experiment with 512x512 (which will make it so that the models process 1 patch instead of 4).
# - We add a colorbar to NDVI, LSWI, and EVI images, it has a height of 100px. So the image becomes 768x868.
W_DIMENSION = 512
H_DIMENSION = 512
DIMENSIONS = f"{W_DIMENSION}x{H_DIMENSION}"


def get_closest_river(ee_polygon: ee.Geometry) -> dict | None:
    """
    Calculate the distance *and* location of the closest medium-sized river to a polygon.

    The search considers rivers contained in the "WWF/HydroSHEDS/v1/FreeFlowingRivers" dataset
    with river orders 4-7 (medium-sized rivers suitable for terra preta sites) that intersect
    a 5 km buffer around the supplied polygon.

    Args:
        ee_polygon (ee.Geometry): Region of interest.

    Returns:
        dict | None: ``None`` if no suitable river is found within the 5 km buffer. Otherwise a dictionary with the following keys:

        - ``distance_m`` – *float* minimal distance in metres between the cell centroid and the nearest river segment.
        - ``longitude`` – *float* longitude of the closest point (decimal degrees).
        - ``latitude`` – *float* latitude of the closest point (decimal degrees).
        - ``river_order`` – *int* river order (4-7).
    """
    centroid = ee_polygon.centroid()
    search_buffer = ee_polygon.buffer(5000)  # 5 km buffer
    rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
    # Filter by river order 4-7 (medium-sized rivers) and bounds
    river_order_filter = ee.Filter.inList("RIV_ORD", [4, 5, 6, 7])
    rivers_nearby = rivers.filter(river_order_filter).filterBounds(search_buffer)

    # Annotate each nearby river with its distance to the polygon centroid.
    def add_distance(feature: ee.Feature) -> ee.Feature:  # type: ignore[valid-type]
        distance = feature.geometry().distance(centroid)
        river_order = feature.get("RIV_ORD")
        return feature.set("distance_m", distance).set("river_order", river_order)

    rivers_with_distance = rivers_nearby.map(add_distance)
    # Find the minimum distance.
    min_distance_ee = rivers_with_distance.aggregate_min("distance_m")
    min_distance = min_distance_ee.getInfo()
    # If no river falls within the buffer, return None.
    if min_distance is None:
        return None
    # Identify the river feature that is closest to the centroid.
    closest_river = ee.Feature(rivers_with_distance.sort("distance_m").first())
    closest_river_geometry = closest_river.geometry()
    closest_point = centroid.closestPoint(closest_river_geometry)
    coords = closest_point.getInfo()
    river_order = closest_river.get("river_order").getInfo()
    lon, lat = (
        coords["coordinates"][0],
        coords["coordinates"][1],
    )
    return {
        "distance_m": round(float(min_distance), 3),
        "longitude": round(float(lon), 3),
        "latitude": round(float(lat), 3),
        "river_order": int(river_order),
    }


def get_precipitation_report(ee_polygon: ee.Geometry, year: int) -> dict:
    """
    Retrieve cumulative precipitation data for a polygon by month for a given year.

    Returns:
        dict: A dictionary with the following keys:
            - "month": The month of the year.
            - "precipitation": The cumulative precipitation for the month.
    """

    def add_month_year(image):
        date = image.date()
        year = date.get("year")
        month = date.get("month")
        return (
            image.set("year", year)
            .set("month", month)
            .set(
                "year_month",
                ee.String(year.format("%04d")).cat("-").cat(month.format("%02d")),
            )
        )

    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(ee_polygon)
        .map(add_month_year)
    )

    def get_monthly_total(year_month):
        monthly_images = chirps.filter(ee.Filter.eq("year_month", year_month))
        monthly_sum = monthly_images.sum()

        precipitation = monthly_sum.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_polygon,
            scale=5566,
            maxPixels=1e9,
        )

        return ee.Feature(
            None,
            {
                "year_month": year_month,
                "precipitation": precipitation.get("precipitation"),
            },
        )

    year_months = chirps.aggregate_array("year_month").distinct().sort()
    monthly_precipitation = ee.FeatureCollection(year_months.map(get_monthly_total))
    monthly_data = monthly_precipitation.getInfo()

    result = {}
    for feature in monthly_data["features"]:
        props = feature["properties"]
        if props["precipitation"] is not None:
            month_num = props["year_month"].split("-")[1]
            month_names = {
                "01": "January",
                "02": "February",
                "03": "March",
                "04": "April",
                "05": "May",
                "06": "June",
                "07": "July",
                "08": "August",
                "09": "September",
                "10": "October",
                "11": "November",
                "12": "December",
            }
            month = month_names[month_num]
            result[month] = round(props["precipitation"], 3)

    return result


SENTINEL_2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# Pixel-scale (metres) to use when computing stats for each index.
#
# – NDVI and EVI are built only from the 10 m resolution bands (B8, B4, B2),
#   so we keep their statistics at 10 m.
# – IRECI mixes a 10 m band (B4) with four 20 m red-edge bands (B5, B6, B7,
#   B8A).  Google Earth Engine will automatically upsample/downsample to the
#   coarsest native resolution invoked in an expression; here that is 20 m.  We
#   therefore request statistics at 20 m to avoid unnecessarily resampling the
#   red-edge inputs.
#
# If you ever switch to a different index that relies exclusively on 20 m bands
# (e.g. MSI SWIR indices) you can reuse `SWIR_SCALE`.

SENTINEL_2_SCALE = 10  # 10 m bands (B8, B4, B2) – NDVI & EVI
SWIR_SCALE = 20  # 20 m bands (B5, B6, B7, B8A, B11) – IRECI and SWIR indices
CLOUDY_PIXEL_PERCENTAGE = 20
MIN = -1
MAX = 1
NDVI_VIZ = {
    "min": MIN,
    "max": MAX,
    "palette": [
        "#000066",  # Dark blue (pure water -1.0)
        "#0000CC",  # Medium blue (clean water -0.6)
        "#4169E1",  # Royal blue (muddy water -0.2)
        "#87CEEB",  # Sky blue (shallow/sediment water 0.0)
        "#8B4513",  # Saddle brown (bare soil/very sparse +0.1)
        "#DAA520",  # Goldenrod (sparse vegetation +0.3)
        "#6B8E23",  # Olive (moderate vegetation +0.5)
        "#228B22",  # Forest green (dense forest +0.7)
        "#00FF00",  # Bright green (very dense +0.8)
        "#FFFF00",  # Yellow (anomalous +0.9)
        "#FF6600",  # Orange (terra preta candidates +1.0)
    ],
}
EVI_VIZ = {
    "min": MIN,
    "max": MAX,
    "palette": [
        "#000000",  # Black (no vegetation -1.0)
        "#800000",  # Maroon (bare soil/rock -0.6)
        "#FF4500",  # Orange red (sparse vegetation -0.4)
        "#FFD700",  # Gold (low vegetation -0.2)
        "#FFFF00",  # Yellow (moderate vegetation 0.0)
        "#ADFF2F",  # Green yellow (healthy vegetation +0.2)
        "#00FF00",  # Lime green (dense vegetation +0.4)
        "#00CC00",  # Green (very dense vegetation +0.6)
        "#008000",  # Dark green (forest +0.8)
        "#004000",  # Very dark green (maximum vegetation +1.0)
    ],
}
# IRECI (Inverted Red-Edge Chlorophyll Index)
IRECI_VIZ = {
    "min": MIN,
    "max": 2,
    "palette": [
        "#440154",  # Viridis palette start (dark purple)
        "#482777",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6ece58",
        "#b5de2b",
        "#fde725",  # Bright yellow-green
    ],
}
INDEX_CFG: dict[str, dict] = {
    "ndvi": {  # what we name it in the _add_ndvi function (band name)
        "scale": SENTINEL_2_SCALE,  # metres per pixel for reduceRegion
        "viz": NDVI_VIZ,
    },
    "evi": {
        "scale": SENTINEL_2_SCALE,
        "viz": EVI_VIZ,
    },
    "ireci": {
        "scale": SWIR_SCALE,
        "viz": IRECI_VIZ,
    },
}

stats_reducer = (
    ee.Reducer.minMax()
    .combine(reducer2=ee.Reducer.mean(), sharedInputs=True)
    .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
)


class VegetationIndex:
    def __init__(
        self,
        id: str,
        ee_polygon: ee.Geometry,
        start_date: str,
        end_date: str,
        output_dir: str,
    ):
        self.id = id
        self.ee_polygon = ee_polygon
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir

    def _apply_min_mask(self, composite: ee.Image, min_threshold: float) -> ee.Image:
        """Apply minimum threshold mask to composite image"""
        return (
            composite.updateMask(composite.gte(min_threshold))
            if min_threshold != -1
            else composite
        )

    def _mask_s2_clouds(self, image: ee.Image) -> ee.Image:
        """Masks clouds in a Sentinel-2 image using the QA band.
        Also masks cloud shadow, medium cloud, high cloud, and cirrus using the SCL band.

        Args:        image (ee.Image): A Sentinel-2 image.

        Returns:
            ee.Image: A cloud-masked Sentinel-2 image.
        """
        qa = image.select("QA60")
        scl = image.select("SCL")
        cloud = 1 << 10
        cirrus = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus).eq(0))
            .And(scl.neq(3))  # 3 = cloud shadow
            .And(scl.neq(8))  # 8 = medium cloud
            .And(scl.neq(9))  # 9 = high cloud
            .And(scl.neq(10))  # 10 = cirrus
        )
        # Scale the optical reflectance bands to [0,1] as required for surface-reflectance data. This also divides the integer QA60 and SCL masks, but we never use those bands after this point, so the precision loss is harmless.  If you need the original integer masks later, select only the optical bands before calling ``divide`` and re-attach the masks afterward.
        return image.updateMask(mask).divide(10000)

    def _add_ndvi(self, image: ee.Image) -> ee.Image:
        """Calculate NDVI for a Sentinel-2 image"""
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
        return image.addBands(ndvi)

    def _add_evi(self, image: ee.Image) -> ee.Image:
        """Calculate EVI for a Sentinel-2 image using NIR, Red, and Blue bands

        EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))

        Uses Sentinel-2 bands:
        - NIR: B8 (842nm)
        - Red: B4 (665nm)
        - Blue: B2 (490nm)
        """
        nir = image.select("B8")
        red = image.select("B4")
        blue = image.select("B2")

        # EVI formula: 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
        evi = image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {"NIR": nir, "RED": red, "BLUE": blue},
        ).rename("evi")

        return image.addBands(evi)

    def _add_ireci(self, image: ee.Image) -> ee.Image:
        """Calculate IRECI for a Sentinel-2 image.

        Formula (Sentinel-2):
            IRECI = (B7 - B4) / (B5 / B6)

        Bands:
            B7 (783 nm), B6 (740 nm), B5 (705 nm) at 20 m; B4 (665 nm) at 10 m.
        """

        ireci = image.expression(
            "(B7 - B4) / (B5 / B6)",
            {
                "B7": image.select("B7"),
                "B6": image.select("B6"),
                "B5": image.select("B5"),
                "B4": image.select("B4"),
            },
        ).rename("ireci")

        return image.addBands(ireci)

    def _add_colorbar_to_image(self, image_path: str, index_name: str) -> None:
        """
        Add colorbar to an existing vegetation index image.

        Args:
            image_path: Path to the image file to modify
            index_name: Name of the vegetation index (for colorbar styling)
        """
        colorbar_height = 100
        gap = 10

        def _create_colorbar_array(colorbar_width: int) -> np.ndarray:
            """Create colorbar as numpy array"""
            viz_params = INDEX_CFG[index_name]["viz"]
            min_val = viz_params["min"]
            max_val = viz_params["max"]
            cmap = mcolors.LinearSegmentedColormap.from_list(
                index_name.upper(), viz_params["palette"], N=256
            )

            # Create figure
            fig = plt.figure(
                figsize=(colorbar_width / 100, colorbar_height / 100), dpi=100
            )
            ax = fig.add_axes([0.05, 0.4, 0.9, 0.3])

            # Create gradient
            gradient = np.linspace(0, 1, colorbar_width).reshape(1, -1)
            ax.imshow(
                gradient,
                aspect="auto",
                cmap=cmap,
                extent=[min_val, max_val, 0, 1],
            )

            # Customize ticks
            tick_positions = np.linspace(min_val, max_val, 11)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [f"{x:.2f}" for x in tick_positions],
                fontsize=9,
                fontweight="bold",
                color="black",
            )
            ax.set_yticks([])

            # Style
            ax.tick_params(
                axis="x", which="both", length=6, width=1.5, direction="out", pad=5
            )

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("black")

            # Add title with appropriate index name
            title_text = f"{index_name.upper()} {self.start_date} - {self.end_date} Values ({min_val} - {max_val})"

            fig.text(
                0.5,
                0.85,
                title_text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

            # Convert to array
            buf = io.BytesIO()
            plt.savefig(
                buf,
                format="jpg",
                dpi=100,
                facecolor="white",
                edgecolor="none",
                bbox_inches="tight",
                pad_inches=0.05,
            )
            buf.seek(0)
            colorbar_img = Image.open(buf)
            plt.close()

            colorbar_img = colorbar_img.resize(
                (colorbar_width, colorbar_height), Image.Resampling.LANCZOS
            )

            return np.array(colorbar_img)

        # Load the original image
        index_img = Image.open(image_path)
        image_width, image_height = index_img.size

        # Create colorbar
        colorbar_array = _create_colorbar_array(image_width)
        colorbar_img = Image.fromarray(colorbar_array)

        # Create combined image
        combined_height = image_height + gap + colorbar_height
        combined_img = Image.new("RGB", (image_width, combined_height), "white")
        combined_img.paste(index_img, (0, 0))
        combined_img.paste(colorbar_img, (0, image_height + gap))

        # Save the combined image (overwrite original)
        combined_img.save(image_path, "JPEG", quality=95)

    def _download_and_combine_index_with_colorbar(
        self,
        index_info_list: list[dict[str, str]],
    ) -> list[dict[str, dict[str, str | bool]]]:
        """
        Download and combine vegetation index images with colorbars in parallel.

        Args:
            index_info_list: List of dicts, each containing:
                - "index_name": str
                - "url": str
                - "output_path": str

        Returns:
            List of dicts with:
                - "index_name": str
                - "output_path": str
                - "success": bool
        """

        def download_and_process_single(
            index_info: dict[str, str],
        ) -> dict[str, str | bool]:
            index_name = index_info["index_name"]
            url = index_info["url"]
            output_path: Path = Path(index_info["output_path"])

            try:
                # timeout is 5 minutes
                response = requests.get(url, timeout=300)
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(response.content)
                download_success = True
            except Exception as e:
                print(f"Failed to download {url} to {output_path}: {e}")
                return {
                    "index_name": index_name,
                    "output_path": str(output_path),
                    "success": False,
                }

            # If download succeeded, add colorbar
            if download_success:
                try:
                    self._add_colorbar_to_image(str(output_path), index_name)
                    return {
                        "index_name": index_name,
                        "output_path": str(output_path),
                        "success": True,
                    }
                except Exception as e:
                    print(f"Failed to add colorbar to {output_path}: {e}")
                    return {
                        "index_name": index_name,
                        "output_path": str(output_path),
                        "success": False,
                    }

            return {
                "index_name": index_name,
                "output_path": str(output_path),
                "success": False,
            }

        # Process all indices in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(download_and_process_single, index_info)
                for index_info in index_info_list
            ]
            results = [future.result() for future in futures]

        return results

    def get_vegetation_indices_for_polygon(
        self,
    ) -> dict[str, dict]:
        """
        Get vegetation indices for a polygon
        Computes NDVI, EVI, and IRECI for a polygon.

        Returns:
            dict[str, dict]: A dictionary with the following keys:
                - "ndvi": {
                    "url": The URL of the NDVI image.
                    "stats": The statistics of the NDVI.
                }
                - "evi": {
                    "url": The URL of the EVI image.
                    "stats": The statistics of the EVI.
                }
                - "ireci": {
                    "url": The URL of the IRECI image.
                    "stats": The statistics of the IRECI.
                }
        """

        def _prep(img):
            # 1) Cloud-mask & scale reflectance
            img = self._mask_s2_clouds(img)
            # 2) Clip early so all further ops run only on the 1 km² ROI
            img = img.clip(self.ee_polygon)
            # 3) Drop QA/SCL masks now – keep only raw spectral bands needed downstream
            img = img.select(["B2", "B4", "B5", "B6", "B7", "B8"])
            # 4) Derive the vegetation indices
            img = self._add_ndvi(img)
            img = self._add_evi(img)
            img = self._add_ireci(img)
            # 5) Keep only the index bands; discard raw bands to lighten later ops
            return img.select(["ndvi", "evi", "ireci"])

        # Build the Sentinel-2 image collection, apply per-image prep, but defer the median
        s2_collection = (
            ee.ImageCollection(SENTINEL_2_COLLECTION)
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.ee_polygon)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUDY_PIXEL_PERCENTAGE))
            .select(
                ["B2", "B4", "B5", "B6", "B7", "B8", "QA60", "SCL"]
            )  # QA/SCL needed for cloud mask
            .map(_prep)
        )

        # Exit early if no images intersect the ROI/date-range
        if s2_collection.size().getInfo() == 0:
            print(
                f"{self.id} - No images found for {self.start_date} - {self.end_date}"
            )
            return None

        # Median composite (already clipped & trimmed to 3 bands)
        composite = s2_collection.median().clip(self.ee_polygon)

        results = {}
        for index_name, cfg in INDEX_CFG.items():
            band_img = self._apply_min_mask(
                composite.select(index_name), cfg["viz"]["min"]
            )
            stats = band_img.reduceRegion(
                reducer=stats_reducer,
                geometry=self.ee_polygon,
                scale=cfg["scale"],
            ).getInfo()
            index_viz = band_img.visualize(**cfg["viz"])
            url_params = {
                "region": self.ee_polygon,
                "dimensions": DIMENSIONS,
                "format": "jpg",
            }
            url = index_viz.getThumbUrl(url_params)
            results[index_name] = {
                "url": url,
                "stats": stats,
            }
        # lets round all the stats to 2 decimal places
        for index_name, result in results.items():
            result["stats"] = {k: round(v, 2) for k, v in result["stats"].items()}
        return results

    def download_and_process_vegetation_indices(
        self, results: dict[str, dict]
    ) -> dict[str, dict[str, str | bool]]:
        """
        Download and process vegetation indices for a polygon.

        Args:
            results: Dict with index names as keys and dicts containing "url" and "stats" as values

        Returns:
            Dict with index names as keys and dicts containing:
                - "output_path": str - path to the processed image
                - "success": bool - whether download and processing succeeded
        """
        # Build list of index info for the combined download/colorbar method
        index_info_list = []
        for index_name, result in results.items():
            url = result["url"]
            output_path = (
                Path(self.output_dir)
                / f"{index_name}_{self.start_date}-{self.end_date}.jpg"
            )
            index_info_list.append(
                {
                    "index_name": index_name,
                    "url": url,
                    "output_path": output_path,
                }
            )

        # Download and process all indices
        processing_results = self._download_and_combine_index_with_colorbar(
            index_info_list
        )

        # Convert results to the expected format
        final_results = {}
        for result in processing_results:
            index_name = result["index_name"]
            final_results[index_name] = {
                "output_path": result["output_path"],
                "success": result["success"],
            }

        return final_results


# ------------------------------------------------------------------------------------------------
# Map generation functions
# ------------------------------------------------------------------------------------------------


def generate_nearest_neighbors_map(
    id: str,
    centroid: tuple[float, float],
    ee_polygon: ee.Geometry,
    neighbors: dict[str, Polygon],
    output_path: str,
):
    m = _get_base_map(centroid, 14)
    neighbor_style = {"color": "000000ff", "width": 4, "fillColor": "00000000"}
    # Convert Shapely polygons to Earth Engine geometries.
    neighbor_ee_features = []
    for neighbor_id, polygon in neighbors.items():
        try:
            ee_geom = ee.Geometry.Polygon(list(polygon.exterior.coords))
            neighbor_ee_features.append(ee.Feature(ee_geom, {"cell_id": neighbor_id}))
        except Exception as e:
            # Skip invalid geometries but log for later inspection.
            print(
                f"[WARN] Skipping neighbour '{neighbor_id}' due to invalid geometry: {e}"
            )

    if not neighbor_ee_features:
        raise ValueError(
            "No valid neighbour geometries were generated – cannot build neighbours map."
        )

    neighbor_features = ee.FeatureCollection(neighbor_ee_features)
    m.addLayer(neighbor_features.style(**neighbor_style), {}, "Grid Cell Boundary")
    m.add_labels(
        data=neighbor_features,
        column="cell_id",
        font_size="10pt",
        font_color="black",
    )
    m = _add_marker_and_boundary(m, centroid)
    _export_map_to_jpg_with_selenium(m, output_path, id)


def generate_closest_river_map(
    id: str,
    centroid: tuple[float, float],
    ee_polygon: ee.Geometry,
    closest_river: dict,
    output_path: str,
):
    distance_km = closest_river["distance_m"] / 1000
    zoom_level = max(8, min(15, int(15 - distance_km / 10)))
    dataset = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
    dataset_vis = ee.Image().byte().paint(dataset, "RIV_ORD", 2)
    vis_params = {
        "min": 1,
        "max": 10,
        "palette": ["08519c", "3182bd", "6baed6", "bdd7e7", "eff3ff"],
    }
    m = _get_base_map(centroid, zoom_level)
    m.addLayer(dataset_vis, vis_params, "Free flowing rivers")
    m = _add_marker_and_boundary(m, centroid)
    folium.Marker(
        location=[closest_river["latitude"], closest_river["longitude"]],
        tooltip="Closest River",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)
    line = ee.Geometry.LineString(
        [
            [centroid[0], centroid[1]],
            [closest_river["longitude"], closest_river["latitude"]],
        ]
    )
    m.addLayer(
        line,
        {"color": "orange", "width": 2},
        f"Distance: {closest_river['distance_m']:.0f}m",
    )
    _export_map_to_jpg_with_selenium(m, output_path, id)


def generate_mapbox_map(
    centroid: tuple[float, float],
    ee_polygon: ee.Geometry,
    output_path: str,
):
    # Earth-Engine returns bbox as a linear-ring polygon: [[[lon, lat], ...]].
    # Flatten it to minLon,minLat,maxLon,maxLat for Mapbox.
    raw_coords = ee_polygon.bounds().getInfo()["coordinates"][0]
    lons = [c[0] for c in raw_coords]
    lats = [c[1] for c in raw_coords]
    minlon, maxlon = min(lons), max(lons)
    minlat, maxlat = min(lats), max(lats)

    bbox_str = f"[{minlon},{minlat},{maxlon},{maxlat}]"

    url = (
        "https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/static/"
        f"pin-l+ff00dd({centroid[0]},{centroid[1]})/"
        f"{bbox_str}/{W_DIMENSION}x{H_DIMENSION}?access_token={settings.MAPBOX_API_KEY}"
    )

    response = requests.get(url)
    response.raise_for_status()

    output_path = Path(output_path)

    # It downloads PNG; convert to JPEG.
    output_path_png = output_path.with_suffix(".png")
    try:
        with open(output_path_png, "wb") as f:
            f.write(response.content)
        img = Image.open(output_path_png)
        img.save(output_path, "JPEG")
    finally:
        if output_path_png.exists():
            output_path_png.unlink()


def _get_base_map(centroid: tuple[float, float], zoom_level: int) -> geemap.Map:
    return geemap.Map(
        center=[centroid[1], centroid[0]],
        zoom=zoom_level,
        attributionControl=False,
        zoom_control=False,
        plugin_LayerControl=False,
        plugin_Fullscreen=False,
        plugin_Draw=False,
        search_control=False,
    )


def _add_marker_and_boundary(
    m: geemap.Map, centroid: tuple[float, float]
) -> geemap.Map:
    folium.Marker(
        location=[centroid[1], centroid[0]],
        tooltip=f"Grid Cell {centroid[0]:.2f}, {centroid[1]:.2f}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)
    style = {"color": "000000ff", "width": 4, "fillColor": "00000000"}
    fc = ee.FeatureCollection(ee.Geometry.Point(centroid))
    m.addLayer(fc.style(**style), {}, "Grid Cell Boundary")
    return m


def _export_map_to_jpg_with_selenium(
    map_obj: geemap.Map,
    output_path: str,
    id: str,
):
    """
    Export a map to a JPG file using Selenium.

    We increase the height by 139px to account for this:
    https://github.com/mozilla/geckodriver/issues/1744

    Args:
        map_obj (geemap.Map): The map object to export.
        output_path (str): The path to save the map to.
    """
    height = H_DIMENSION + 139

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={W_DIMENSION},{height}")
    options.add_argument("--force-device-scale-factor=1")

    # Use a unique filename to avoid clashes when multiple cells are processed in parallel.
    html_path = Path("tmp") / f"temp_map_{id}_{uuid.uuid4().hex}.html"

    map_obj.to_html(str(html_path))
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_window_size(W_DIMENSION, height)
        driver.get(f"file://{html_path.resolve()}")
        time.sleep(3)
        driver.save_screenshot(str(output_path))
    finally:
        if driver is not None:
            driver.quit()
        if html_path.exists():
            html_path.unlink()


class ExperimentManager:
    def __init__(
        self,
        experiment_id: str,
        target_cell_id: str,
        target_cell_polygon: Polygon,
        neighbors: dict[str, Polygon],
    ):
        self.target_cell_id = target_cell_id
        self.target_cell_polygon = target_cell_polygon
        self.target_cell_centroid = (
            target_cell_polygon.centroid.x,
            target_cell_polygon.centroid.y,
        )
        self.target_cell_ee_polygon = ee.Geometry.Polygon(
            list(target_cell_polygon.exterior.coords)
        )
        self.neighbors = neighbors

        # Each target cell gets its own sandboxed directory: tmp/<experiment_id>/<target_cell_id>/
        self.experiment_id = experiment_id
        base_dir = Path("tmp") / experiment_id / self.target_cell_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # Directory for the target cell itself (i.e., summary maps)
        self.target_cell_output_dir = base_dir

        # Directories for neighbour cells live one level below the target cell dir
        self.cell_id_2_output_dir = {}
        for cell_id, polygon in self.neighbors.items():
            cell_output_dir = base_dir / cell_id
            cell_output_dir.mkdir(parents=True, exist_ok=True)
            self.cell_id_2_output_dir[cell_id] = cell_output_dir

    def _with_cache(
        self, cache_name: str, args: dict[str, Any], compute_fn: Callable[[], dict]
    ) -> dict:
        """Generic disk-based cache wrapper.

        Args:
            cache_name: A short name grouping similar computations (e.g. 'vegetation_index').
            args: Dict of arguments that uniquely identify the computation.  Must be JSON-serialisable or convertible via ``str``.
            compute_fn: Callable that performs the expensive operation and returns the usual ``{"response": str, "images": list[PIL.Image]}`` structure.

        Returns
        -------
        Same structure as ``compute_fn`` but loaded from cache when available.
        """
        key_str = json.dumps(args, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        cache_dir = Path("tmp") / "cache" / cache_name / key_hash
        output_json = cache_dir / "output.json"

        # Try cache hit -------------------------------------------------------
        if output_json.exists():
            try:
                with output_json.open() as f:
                    meta = json.load(f)
                images: list[Image.Image] = []
                for img_file in meta.get("images", []):
                    img_path = cache_dir / img_file
                    if img_path.exists():
                        images.append(Image.open(img_path))
                return {"response": meta["response"], "images": images}
            except Exception as e:
                # Fall through to recompute on any error (corrupt cache etc.)
                print(f"[CACHE] Failed to load cache for {cache_name}: {e}")

        # Cache miss ----------------------------------------------------------
        result = compute_fn()
        try:
            lock = FileLock(str(cache_dir / ".lock"))
            with lock:
                cache_dir.mkdir(parents=True, exist_ok=True)
                image_files: list[str] = []
                for idx, img in enumerate(result.get("images", [])):
                    img_filename = f"img_{idx}.jpg"
                    img_path = cache_dir / img_filename
                    try:
                        img.save(img_path, "JPEG")
                        image_files.append(img_filename)
                    except Exception as e:
                        print(f"[CACHE] Could not save image for {cache_name}: {e}")
                with output_json.open("w") as f:
                    json.dump(
                        {"response": result["response"], "images": image_files}, f
                    )
        except Exception as e:
            print(f"[CACHE] Failed to write cache for {cache_name}: {e}")

        return result

    def get_closest_river(self) -> dict:
        args = {"target_cell_id": self.target_cell_id}

        def _compute():
            closest_river_local = get_closest_river(self.target_cell_ee_polygon)
            if closest_river_local is None:
                return {
                    "response": "No river found within 5km of the cell",
                    "images": [],
                }

            output_path_local = self.target_cell_output_dir / "closest_river_map.jpg"
            generate_closest_river_map(
                self.target_cell_id,
                self.target_cell_centroid,
                self.target_cell_ee_polygon,
                closest_river_local,
                output_path_local,
            )
            return {
                "response": json.dumps(closest_river_local),
                "images": [Image.open(output_path_local)],
            }

        return self._with_cache("closest_river", args, _compute)

    def get_precipitation_report(self, year: int) -> dict:
        args = {"target_cell_id": self.target_cell_id, "year": year}

        def _compute():
            from src.evals.terra_preta import tools as _tp  # avoid name shadowing

            precipitation_report_local = _tp.get_precipitation_report(
                self.target_cell_ee_polygon, year
            )
            return {
                "response": json.dumps(precipitation_report_local),
                "images": [],
            }

        return self._with_cache("precipitation_report", args, _compute)

    def get_mapbox_map(self) -> dict:
        args = {"target_cell_id": self.target_cell_id}

        def _compute():
            output_path_local = self.target_cell_output_dir / "mapbox_map.jpg"
            generate_mapbox_map(
                self.target_cell_centroid,
                self.target_cell_ee_polygon,
                output_path_local,
            )
            return {
                "response": json.dumps({"success": True}),
                "images": [Image.open(output_path_local)],
            }

        return self._with_cache("mapbox_map", args, _compute)

    def get_nearest_neighbors_map(self) -> dict:
        args = {"target_cell_id": self.target_cell_id}

        def _compute():
            output_path_local = (
                self.target_cell_output_dir / "nearest_neighbors_map.jpg"
            )
            generate_nearest_neighbors_map(
                self.target_cell_id,
                self.target_cell_centroid,
                self.target_cell_ee_polygon,
                self.neighbors,
                output_path_local,
            )
            return {
                "response": json.dumps({"success": True}),
                "images": [Image.open(output_path_local)],
            }

        return self._with_cache("nearest_neighbors_map", args, _compute)

    def get_vegetation_index(
        self, cell_id: str, start_date: str, end_date: str
    ) -> dict:
        args = {
            "target_cell_id": self.target_cell_id,
            "cell_id": str(cell_id),
            "start_date": start_date,
            "end_date": end_date,
        }

        def _compute():
            # Normalise the incoming cell identifier to string.
            _cell_id = str(cell_id)

            # Determine which polygon we are dealing with: the target cell itself or a neighbour.
            if _cell_id in (self.target_cell_id, "target"):
                polygon = self.target_cell_polygon
                ee_polygon_local = self.target_cell_ee_polygon
                output_dir_local = self.target_cell_output_dir
            elif _cell_id in self.neighbors:
                polygon = self.neighbors[_cell_id]
                ee_polygon_local = ee.Geometry.Polygon(list(polygon.exterior.coords))
                output_dir_local = self.cell_id_2_output_dir[_cell_id]
            else:
                return {
                    "response": f"Cell id '{_cell_id}' not recognised. It must be either the target cell id ({self.target_cell_id}) or one of the neighbour names: {list(self.neighbors.keys())}.",
                    "images": [],
                }

            vi = VegetationIndex(
                _cell_id, ee_polygon_local, start_date, end_date, output_dir_local
            )
            results = vi.get_vegetation_indices_for_polygon()
            if results is None:
                return {
                    "response": "No Sentinel-2 data found for the given date range for the cell",
                    "images": [],
                }
            downloaded_results = vi.download_and_process_vegetation_indices(results)
            output_local = {}
            images_local = []

            for k, v in downloaded_results.items():
                if v["success"]:
                    output_local[k] = results[k]["stats"]
                    images_local.append(Image.open(v["output_path"]))
                else:
                    output_local[k] = "Failed to download and process vegetation index"
            return {
                "response": json.dumps(output_local),
                "images": images_local,
            }

        return self._with_cache("vegetation_index", args, _compute)

    def process_tool_call(self, r: ResponseOutputItem) -> tuple[dict, dict]:
        tool_name = r.name
        args = json.loads(r.arguments)
        call_id = r.call_id

        if tool_name == "finish_analysis":
            return None, None

        if tool_name == "get_precipitation_report":
            result = self.get_precipitation_report(args["year"])
        elif tool_name == "get_vegetation_indices":
            result = self.get_vegetation_index(
                args["cell_id"], args["start_date"], args["end_date"]
            )
        else:
            raise ValueError(f"Unknown tool name: {tool_name}")
        fn_output = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": result["response"],
        }

        user_reply = None
        if result["images"]:
            content = []
            for img in result["images"]:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encode_image_pil(img)}",
                        "detail": "high",
                    }
                )
            user_reply = {
                "role": "user",
                "content": content,
            }
        return fn_output, user_reply
