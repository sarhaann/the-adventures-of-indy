from pathlib import Path

from src.config import settings
from src.scripts.lidar_tiles.prepare_eval_data import _get_all_lidar_tiles


def get_missing_zscore_keys() -> list[str]:
    """Return LAZ/LAS keys that still lack z-score images, *excluding* parent
    transect files for which split-tile outputs already exist.
    """

    # ------------------------------------------------------------------
    # 1. Directories that already contain a z-score image.
    # ------------------------------------------------------------------
    processed_dirs: set[str] = set()
    paginator = settings.r2_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.R2_BUCKET_NAME):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("zscore.jpg"):
                processed_dirs.add(str(Path(key).parent))

    # ------------------------------------------------------------------
    # 2. Scan all LAZ/LAS keys and simultaneously collect potential parent keys
    #    that we should ignore because split-tiles exist.
    # ------------------------------------------------------------------
    laz_keys: list[str] = []
    split_parent_candidates: set[str] = set()

    paginator = settings.r2_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.R2_BUCKET_NAME):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.lower().endswith((".laz", ".las")):
                laz_keys.append(key)

                # Detect split-tile pattern "…/<stem>_tile_###/<stem>_tile_###.laz"
                p = Path(key)
                if "_tile_" in p.stem:
                    grandparent = p.parent.parent  # eba/<transect_id>
                    stem_prefix = p.stem.split("_tile_")[0]
                    split_parent_candidates.add(
                        str(grandparent / f"{stem_prefix}{p.suffix}")
                    )

    print(f"Found {len(laz_keys):,} LAZ/LAS tiles in the bucket.")

    missing_keys = [
        k
        for k in laz_keys
        if str(Path(k).parent) not in processed_dirs
        and k not in split_parent_candidates
    ]

    # ------------------------------------------------------------------
    # 3. Keep only keys that exist in the authoritative tile manifest
    #    returned by _get_all_lidar_tiles() to avoid malformed or duplicate
    #    paths.
    # ------------------------------------------------------------------
    try:
        allowed_keys: set[str] = {
            tile["properties"]["key"] for tile in _get_all_lidar_tiles()
        }
        pre_filter_count = len(missing_keys)
        missing_keys = [k for k in missing_keys if k in allowed_keys]
        if len(missing_keys) != pre_filter_count:
            print(
                f"Filtered out {pre_filter_count - len(missing_keys):,} keys not present in _get_all_lidar_tiles()."
            )
    except Exception as e:  # pragma: no cover – defensive; continue if manifest fails
        print(f"Warning: could not import _get_all_lidar_tiles(): {e}")

    # Deduplicate while preserving original order
    seen: set[str] = set()
    missing_keys = [k for k in missing_keys if not (k in seen or seen.add(k))]

    print(f"Tiles still missing z-scores: {len(missing_keys):,}")
    return missing_keys
