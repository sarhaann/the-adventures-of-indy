from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import fire
from rich import print
from rich.console import Console
from rich.table import Table

__all__ = ["compute_stats", "print_stats"]


def _extract_function_args(output: Dict[str, Any]) -> dict | None:
    """Return the first function_call argument object inside an OpenAI response."""
    for o in output.get("output", []):
        if o.get("type") == "function_call":
            try:
                return json.loads(o["arguments"])
            except (KeyError, json.JSONDecodeError):
                return None
    return None


def compute_stats(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate confidence scores and token counts across result subsets.

    The subset (confirmed / control / test) for each tile is looked up from
    ``data/evals/lidar_tiles_signed_urls.json`` so callers only need to supply
    the *results* mapping produced by ``run_eval.py``.
    """

    # Build tile_key → subset mapping ------------------------------------
    try:
        with open("data/evals/lidar_tiles_signed_urls.json", "r") as f:
            tile_info: List[dict] = json.load(f)
        subset_lookup: Dict[str, str] = {t["tile_key"]: t["subset"] for t in tile_info}
    except FileNotFoundError:
        subset_lookup = {}

    # Prepare stats container --------------------------------------------
    stats: Dict[str, Any] = {
        "confirmed": {},
        "control": {},
        "test": {},
        "unknown": {},
        "total_tokens": 0,
    }

    for key, result in results.items():
        output = result.get("output")
        if output is None:
            # Skip failed runs.
            continue

        # Token usage
        usage = output.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        stats["total_tokens"] += total_tokens

        # Extract function-call arguments
        args = _extract_function_args(output)
        if args is None:
            continue

        subset = subset_lookup.get(key, "unknown")

        stats[subset][key] = {
            "analysis": args.get("analysis"),
            "confidence_score": args.get("confidence_score"),
            "tokens": total_tokens,
        }

    return stats


def print_stats(stats: Dict[str, Any]) -> None:  # noqa: D401 – simple imperative name
    """Pretty-print statistics using Rich tables."""
    console = Console()

    subsets = ["confirmed", "control", "test"]
    table = Table(title="LiDAR Tile Evaluation Stats", show_lines=True)
    table.add_column("Subset", style="cyan", no_wrap=True)
    table.add_column("# Tiles", justify="right")
    table.add_column("Avg Confidence", justify="right")
    table.add_column("Avg Tokens", justify="right")

    # Populate rows
    for subset in subsets:
        entries = list(stats[subset].values())
        count = len(entries)
        if count:
            avg_conf = sum(e["confidence_score"] for e in entries) / count
            avg_tokens = sum(e["tokens"] for e in entries) / count
            table.add_row(subset, str(count), f"{avg_conf:.2f}", f"{avg_tokens:.0f}")
        else:
            table.add_row(subset, "0", "-", "-")

    console.print(table)

    # ------------------------------------------------------------------
    # Additional statistics: max/min scores and threshold counts
    # ------------------------------------------------------------------
    threshold = 8  # Scores above this value are considered a probable site
    summary_table = Table(
        title="Confidence Score Extremes and Threshold Counts",
        show_lines=True,
    )
    summary_table.add_column("Subset", style="cyan", no_wrap=True)
    summary_table.add_column("Max Score", justify="right")
    summary_table.add_column("Min Score", justify="right")
    summary_table.add_column(f"# > {threshold}", justify="right")
    summary_table.add_column(f"% > {threshold}", justify="right")

    for subset in subsets:
        entries = list(stats[subset].values())
        # Filter out None scores, if any
        scores = [
            e["confidence_score"] for e in entries if e["confidence_score"] is not None
        ]
        if scores:
            max_score = max(scores)
            min_score = min(scores)
            gt_threshold = sum(1 for s in scores if s > threshold)
            percent_threshold = (gt_threshold / len(scores)) * 100
            summary_table.add_row(
                subset,
                f"{max_score:.2f}",
                f"{min_score:.2f}",
                str(gt_threshold),
                f"{percent_threshold:.1f}%",
            )
        else:
            summary_table.add_row(subset, "-", "-", "-", "-")

    console.print(summary_table)

    # ------------------------------------------------------------------
    # Overall token statistics
    # ------------------------------------------------------------------
    total_results = sum(len(stats[s]) for s in subsets)
    total_tokens = stats["total_tokens"]
    if total_results:
        avg_tokens_overall = total_tokens / total_results
    else:
        avg_tokens_overall = 0

    print(f"[bold]Total tokens:[/] {total_tokens}")
    print(f"[bold]Avg tokens per tile:[/] {avg_tokens_overall:.2f}")


# ---------------------------------------------------------------------------
# Fire CLI entry point
# ---------------------------------------------------------------------------


def main(exp_num: int = 0, exp_type: str = "eval") -> None:  # noqa: D401
    """Compute and display statistics for a given LiDAR experiment.

    Parameters
    ----------
    exp_num
        Experiment number used when running `run_eval.py`.
    exp_type
        Either ``"eval"`` or ``"test"`` (matches `run_eval.py`).
    """

    if exp_type not in {"eval", "test"}:
        raise ValueError("exp_type must be either 'eval' or 'test'")

    results_path = Path(f"data/evals/lidar_eval_output_{exp_type}_{exp_num}.json")

    if not results_path.exists():
        print(
            f"[red]Results file '{results_path}' not found – run the experiment first.[/red]"
        )
        return

    try:
        with results_path.open("r") as f:
            results = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[red]Could not parse JSON in '{results_path}': {e}[/red]")
        return

    stats = compute_stats(results)
    print_stats(stats)

    # ----------------------------------------------------------------------
    # Write GeoJSON output
    # ----------------------------------------------------------------------
    features = _write_geojson(stats, exp_num=exp_num, exp_type=exp_type) or []

    # ----------------------------------------------------------------------
    # Write concise report JSON (confidence_score > 5, sorted desc)
    # ----------------------------------------------------------------------
    _write_report_json(features, exp_num=exp_num, exp_type=exp_type)


# ---------------------------------------------------------------------------
# New helper: write GeoJSON with evaluation results
# ---------------------------------------------------------------------------


def _write_geojson(
    stats: Dict[str, Any], exp_num: int, exp_type: str
) -> list[dict] | None:
    """Create a GeoJSON FeatureCollection with evaluation metadata.

    Each feature from the master LiDAR-tile catalogue is matched by its
    ``key`` to the evaluation *stats*. The resulting feature's properties
    are augmented with the following keys before being written to disk:

    - ``analysis`` – the free-text model analysis string
    - ``confidence_score`` – numeric score returned by the model
    - ``tokens`` – total tokens used for this evaluation call
    - ``subset`` – one of ``confirmed`` / ``control`` / ``test`` / ``unknown``

    The file is saved to
    ``data/evals/lidar_eval_output_{exp_type}_{exp_num}.geojson``.
    """

    try:
        # Lazy import to avoid the costly Shapely import chain when not needed
        from src.scripts.lidar_tiles.prepare_eval_data import (
            _get_all_lidar_tiles,  # type: ignore
        )
    except Exception as e:  # pragma: no cover – log and abort gracefully
        print(
            f"[red]Could not import _get_all_lidar_tiles – skipping GeoJSON generation: {e}[/red]"
        )
        return None

    # Build lookup table: tile_key → feature (copy to avoid in-place mutation)
    try:
        tiles = _get_all_lidar_tiles()
    except Exception as e:  # pragma: no cover – log and abort gracefully
        print(
            f"[red]Failed to obtain LiDAR tiles – skipping GeoJSON generation: {e}[/red]"
        )
        return None

    # ------------------------------------------------------------------
    # Load dataset metadata to map tile keys to their source URL
    # ------------------------------------------------------------------
    try:
        with open("data/lidar_tiles/all_dataset_metadata.json", "r") as f:
            dataset_meta: Dict[str, str] = json.load(f)
    except FileNotFoundError:
        dataset_meta = {}
        print(
            "[yellow]Warning: dataset metadata file not found – 'dataset_url' will be missing.[/yellow]"
        )

    from copy import deepcopy

    tile_lookup = {tile["properties"]["key"]: deepcopy(tile) for tile in tiles if tile}

    output_features: list[dict] = []

    for subset in ("confirmed", "control", "test", "unknown"):
        subset_entries = stats.get(subset, {})
        if not isinstance(subset_entries, dict):
            # Skip non-mapping entries such as the total token counter.
            continue

        for key, info in subset_entries.items():
            feature = tile_lookup.get(key)
            if feature is None:
                print(
                    f"[yellow]Warning: tile with key '{key}' not found in master catalogue – skipping.[/yellow]"
                )
                continue

            # Augment properties with evaluation metadata
            props = feature.get("properties", {})
            props.update(
                {
                    "analysis": info.get("analysis"),
                    "confidence_score": info.get("confidence_score"),
                    "tokens": info.get("tokens"),
                    "subset": subset,
                    "dataset_url": dataset_meta.get(key.split("/", 1)[0]),
                }
            )
            feature["properties"] = props
            output_features.append(feature)

    geojson = {"type": "FeatureCollection", "features": output_features}

    output_path = Path(f"data/evals/lidar_eval_output_{exp_type}_{exp_num}.geojson")
    try:
        with output_path.open("w") as f:
            json.dump(geojson, f)
        print(
            f"[green]→ GeoJSON written to {output_path} ({len(output_features):,} features).[/green]"
        )
    except Exception as e:
        print(f"[red]Failed to write GeoJSON output: {e}[/red]")

    return output_features


# ---------------------------------------------------------------------------
# Helper: write concise report JSON
# ---------------------------------------------------------------------------


def _write_report_json(
    features: List[dict], *, exp_num: int, exp_type: str, threshold: float = 5.0
) -> None:
    """Write a simplified JSON report for high-confidence tiles.

    Parameters
    ----------
    features
        List of GeoJSON Features as returned by `_write_geojson`.
    exp_num, exp_type
        Identifiers used to construct the output filename.
    threshold
        Minimum `confidence_score` required for inclusion.
    """

    # Filter and sort features -------------------------------------------
    high_conf = [
        f
        for f in features
        if (f.get("properties", {}).get("confidence_score") or 0) > threshold
    ]

    high_conf.sort(
        key=lambda f: f["properties"].get("confidence_score", 0), reverse=True
    )

    entries: List[Dict[str, Any]] = []
    for f in high_conf:
        props = f.get("properties", {})
        key_val = props.get("key", "")
        entries.append(
            {
                "dataset_url": props.get("dataset_url"),
                "r2_key": key_val,
                "tile_id": key_val.rsplit("/", 1)[-1],
                "confidence_score": props.get("confidence_score"),
                "analysis": props.get("analysis"),
            }
        )

    output_path = Path(f"data/evals/report_lidar_eval_output_{exp_type}_{exp_num}.json")
    try:
        with output_path.open("w") as f:
            json.dump(entries, f, indent=2)
        print(
            f"[green]→ Report written to {output_path} ({len(entries):,} entries).[/green]"
        )
    except Exception as e:
        print(f"[red]Failed to write report JSON: {e}[/red]")


if __name__ == "__main__":
    fire.Fire(main)
