import json
import os
import random

import numpy as np
from shapely.geometry import shape
from tqdm import tqdm

from src.config import settings  # Provides an authenticated R2 (S3-compatible) client
from src.utils import haversine_np

random.seed(41)
MALFORMED_LIDAR_KEYS = [
    "cms/FST_A01a_2015_laz_5.laz/FST_A01a_2015_laz_5.laz",
    "cms/RIB_A01_2014_laz_7.laz/RIB_A01_2014_laz_7.laz",
    "cms/ST2_A01_2016_laz_12.laz/ST2_A01_2016_laz_12.laz",
]

confirmed_site_datasets = [
    "carolina",
    "coomes",
    "iphan_classified",
    "iriarte",
    "james_q_jacobs_processed",
    "prumers",
    "souza",
    "walker",
    "wilian_processed",
]


def _get_all_confirmed_sites() -> list[dict]:
    """
    returns all sites with confirmed archaeological sites.
    """
    features = []
    for dataset in confirmed_site_datasets:
        with open(f"data/confirmed/{dataset}.geojson", "r") as f:
            data = json.load(f)
            features.extend(data["features"])
    return features


def _get_all_lidar_confirmed_sites() -> list[dict]:
    """
    Only geoglyphs and earthworks sites!

    We get all the confirmed geoglyphs and earthworks sites except wilian, terra preta and 'others' from iphan classified.
    """
    features = []
    for dataset in confirmed_site_datasets:
        if dataset == "wilian_processed":
            continue
        with open(f"data/confirmed/{dataset}.geojson", "r") as f:
            data = json.load(f)["features"]
        if dataset == "iphan_classified":
            data = [
                feature
                for feature in data
                if feature["properties"]["type"]
                in [
                    "high_probability_geoglyphs",
                    "potential_geoglyphs",
                    "high_probability_earthworks",
                    "potential_earthworks",
                ]
            ]
            features.extend(data)
        else:
            features.extend(data)
    return features


def _remove_duplicate_geometries(tiles):
    """Remove tiles with duplicate geometries."""
    seen_geometries = set()
    unique_tiles = []

    for tile in tiles:
        geom = shape(tile["geometry"])
        bounds_key = tuple(round(coord, 6) for coord in geom.bounds)

        if bounds_key not in seen_geometries:
            seen_geometries.add(bounds_key)
            unique_tiles.append(tile)

    return unique_tiles


def _get_all_lidar_tiles() -> list[dict]:
    """
    Get all tiles from all datasets.

    Remove malformed tiles and duplicates.
    """
    tiles = []

    with open("data/lidar_tiles/cms.geojson", "r") as f:
        data = json.load(f)
        tiles.extend(data["features"])

    with open("data/lidar_tiles/nouragues.geojson", "r") as f:
        data = json.load(f)
        tiles.extend(data["features"])

    with open("data/lidar_tiles/others.geojson", "r") as f:
        data = json.load(f)
        tiles.extend(data["features"])

    with open("data/lidar_tiles/paracou.geojson", "r") as f:
        data = json.load(f)
        tiles.extend(data["features"])

    tiles = [
        tile for tile in tiles if tile["properties"]["key"] not in MALFORMED_LIDAR_KEYS
    ]

    original_count = len(tiles)
    tiles = _remove_duplicate_geometries(tiles)
    duplicates_removed = original_count - len(tiles)

    if duplicates_removed > 0:
        print(
            f"Removed {duplicates_removed} duplicate tiles (kept {len(tiles)} unique)"
        )

    return tiles


def _get_lidar_tiles_with_confirmed_sites():
    tiles = _get_all_lidar_tiles()
    confirmed_sites = _get_all_lidar_confirmed_sites()

    # Pre-convert all geometries once (major speedup!)
    tile_geometries = [shape(tile["geometry"]) for tile in tiles]
    site_geometries = [shape(site["geometry"]) for site in confirmed_sites]

    print(f"Checking {len(tiles)} tiles against {len(confirmed_sites)} sites...")

    results = []
    for i, (tile, tile_geom) in enumerate(zip(tiles, tile_geometries)):
        if i % 500 == 0:  # Progress indicator
            print(f"Processed {i}/{len(tiles)} tiles")

        confirmed_sites_in_tile = []
        for site, site_geom in zip(confirmed_sites, site_geometries):
            if tile_geom.intersects(site_geom):
                confirmed_sites_in_tile.append(site)

        if confirmed_sites_in_tile:
            tile["properties"]["confirmed_sites"] = confirmed_sites_in_tile
            results.append(tile)
    with open("data/evals/lidar_tiles_confirmed.json", "w") as f:
        json.dump(results, f)

    print(f"Found {len(results)} tiles with confirmed sites")
    return results


def _get_lidar_tiles_control(n_tiles: int = 20):
    tiles = _get_all_lidar_tiles()
    with open("data/evals/lidar_tiles_confirmed.json", "r") as f:
        confirmed_tiles = json.load(f)
    confirmed_tiles_keys = [tile["properties"]["key"] for tile in confirmed_tiles]
    tiles = [
        tile for tile in tiles if tile["properties"]["key"] not in confirmed_tiles_keys
    ]
    tiles_control = random.sample(tiles, n_tiles)
    with open("data/evals/lidar_tiles_control.json", "w") as f:
        json.dump(tiles_control, f)
    print(f"Found {len(tiles_control)} control tiles")
    return tiles_control


def _get_lidar_test_tiles(
    radius_km: float = 1000.0, n_tiles: int = 1000, return_all: bool = False
):
    """Rank LiDAR tiles by the number of confirmed sites within *radius_km*.

    - Get the top n_tiles by confirmed_site_count
    - Tiles do not include confirmed or control tiles

    - When we run this, we check for the presences of all achaeological sites, even including terra preta sites!

    For each LiDAR tile centre point, count how many confirmed sites lie within
    a great-circle distance of *radius_km*. The resulting list of tile
    ``Feature`` objects is returned in descending order of this count. Each
    tile's ``properties`` dict is augmented with:

    ``confirmed_site_count`` – the integer count of nearby confirmed sites.

    The full, ordered list is also written to
    ``data/evals/lidar_test_tiles.json`` for convenience.

    If return_all is True, return all tiles.
    """
    output_path = "data/evals/lidar_test_tiles.json"
    tiles = _get_all_lidar_tiles()
    with open("data/evals/lidar_tiles_confirmed.json", "r") as f:
        confirmed_tiles = json.load(f)
    with open("data/evals/lidar_tiles_control.json", "r") as f:
        control_tiles = json.load(f)
    confirmed_tiles_keys = [tile["properties"]["key"] for tile in confirmed_tiles]
    control_tiles_keys = [tile["properties"]["key"] for tile in control_tiles]
    tiles = [
        tile for tile in tiles if tile["properties"]["key"] not in confirmed_tiles_keys
    ]
    tiles = [
        tile for tile in tiles if tile["properties"]["key"] not in control_tiles_keys
    ]
    if return_all:
        with open(output_path, "w") as f:
            json.dump(tiles, f)
        return tiles

    confirmed_sites = _get_all_confirmed_sites()
    site_coords = np.array(
        [site["geometry"]["coordinates"] for site in confirmed_sites]
    )
    site_lons = site_coords[:, 0]
    site_lats = site_coords[:, 1]

    tile_centroids = []
    for tile in tiles:
        geom = shape(tile["geometry"])
        centroid = geom.centroid
        tile_centroids.append((centroid.x, centroid.y))

    print(
        f"Computing confirmed-site density for {len(tiles)} tiles using a radius of {radius_km} km…"
    )

    with tqdm(total=len(tiles), desc="Calculating densities") as pbar:
        for tile, (lon, lat) in zip(tiles, tile_centroids):
            # Vectorised distance calculation to all sites
            distances = haversine_np(lon, lat, site_lons, site_lats)
            count = int(np.sum(distances <= radius_km))

            tile.setdefault("properties", {})["confirmed_site_count"] = count
            pbar.update(1)

    # only non zero confirmed site count
    tiles = [tile for tile in tiles if tile["properties"]["confirmed_site_count"] > 0]
    # Sort by descending count
    tiles_sorted = sorted(
        tiles,
        key=lambda t: t["properties"].get("confirmed_site_count", 0),
        reverse=True,
    )
    tiles_sorted = tiles_sorted[:n_tiles]

    # Persist results for easy inspection

    try:
        with open(output_path, "w") as f:
            json.dump(tiles_sorted, f)
        print(f"→ Results written to {output_path}")
    except Exception as e:
        print(f"Warning: could not write results to file – {e}")

    return tiles_sorted


# -----------------------------------------------------------------------------
# Signed-URL generation
# -----------------------------------------------------------------------------


def _ensure_eval_manifests_exist():
    """Ensure that the confirmed, control and test manifests exist on disk.

    If any of the expected ``data/evals`` JSON files are missing, the
    corresponding helper is invoked to (re)generate it.
    """

    confirmed_path = "data/evals/lidar_tiles_confirmed.json"
    control_path = "data/evals/lidar_tiles_control.json"
    test_path = "data/evals/lidar_test_tiles.json"

    if not os.path.exists(confirmed_path):
        print("Confirmed-tile manifest missing – generating…")
        _get_lidar_tiles_with_confirmed_sites()

    if not os.path.exists(control_path):
        print("Control-tile manifest missing – generating…")
        # Use default of 26 tiles (mirrors previous behaviour)
        _get_lidar_tiles_control()

    if not os.path.exists(test_path):
        print("Test-tile manifest missing – generating…")
        _get_lidar_test_tiles()


def _public_url(key: str) -> str:
    """Return a **non-expiring** public URL for a given object *key* in the R2 bucket.

    Assumes that the bucket is configured for public access via the path-style
    endpoint (``<account>.r2.cloudflarestorage.com/<bucket>/<key>``).
    """

    return f"{settings.R2_URL}/{settings.R2_BUCKET_NAME}/{key}"


DEFAULT_EXPIRY_DAYS: int = 7  # R2/S3 maximum (7 days)


def _generate_urls_for_eval_tiles(
    *, use_signed: bool = True, expires_in: int = 60 * 60 * 24 * DEFAULT_EXPIRY_DAYS
):
    """Generate URLs for ``dem.jpg`` and ``zscore.jpg`` for all evaluation tiles.

    Parameters
    ----------
    use_signed:
        If ``True`` (default), create presigned URLs that expire after
        *expires_in* seconds. The default is **7 days (604 800 s)** – the
        maximum allowed by Cloudflare R2's S3-compatible API. Attempting to
        exceed this limit will raise a `SignatureDoesNotMatch` error.
    expires_in:
        Lifetime of the presigned links in seconds (only used when
        ``use_signed`` is ``True``).
    """

    _ensure_eval_manifests_exist()

    subset_files = {
        "confirmed": "data/evals/lidar_tiles_confirmed.json",
        "control": "data/evals/lidar_tiles_control.json",
        "test": "data/evals/lidar_test_tiles.json",
    }

    r2 = settings.r2_client if use_signed else None
    bucket = settings.R2_BUCKET_NAME

    entries: list[dict] = []

    for subset, path in subset_files.items():
        try:
            with open(path, "r") as f:
                tiles = json.load(f)
        except FileNotFoundError:
            print(
                f"Warning: expected manifest {path} not found – skipping {subset} subset."
            )
            continue

        print(
            f"Generating {'signed' if use_signed else 'public'} URLs for {len(tiles):,} {subset} tiles…"
        )

        for tile in tiles:
            laz_key: str = tile["properties"]["key"]
            parent_dir = laz_key.rsplit("/", 1)[0]

            dem_key = f"{parent_dir}/dem.jpg"
            zscore_key = f"{parent_dir}/zscore.jpg"

            if use_signed:

                def _signed(k: str):
                    try:
                        return r2.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": bucket, "Key": k},
                            ExpiresIn=expires_in,
                        )
                    except Exception as e:  # pragma: no cover
                        print(f"  • Failed to sign {k}: {e}")
                        return None

                dem_url = _signed(dem_key)
                zscore_url = _signed(zscore_key)
            else:
                dem_url = _public_url(dem_key)
                zscore_url = _public_url(zscore_key)

            entries.append(
                {
                    "subset": subset,
                    "tile_key": laz_key,
                    "dem_url": dem_url,
                    "zscore_url": zscore_url,
                }
            )

    suffix = "signed" if use_signed else "public"
    output_path = f"data/evals/lidar_tiles_{suffix}_urls.json"
    with open(output_path, "w") as f:
        json.dump(entries, f)

    if use_signed:
        print(
            f"→ Wrote signed URLs for {len(entries):,} tiles to {output_path} (valid for {expires_in / 3600:.1f} h)."
        )
    else:
        print(f"→ Wrote public URLs for {len(entries):,} tiles to {output_path}.")

    return entries


if __name__ == "__main__":
    _get_lidar_tiles_with_confirmed_sites()
    _get_lidar_tiles_control()
    _get_lidar_test_tiles(return_all=True)
    # Generate signed URLs that expire in 7 days (R2 maximum).
    _generate_urls_for_eval_tiles(
        use_signed=True, expires_in=60 * 60 * 24 * DEFAULT_EXPIRY_DAYS
    )
