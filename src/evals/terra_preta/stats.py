import argparse
import json
import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.evals.terra_preta.data import (
    get_confirmed_and_control_sites_for_terra_preta,
)

console = Console()


def _compute_stats(scores: list[int]) -> dict[str, float]:
    """Return min, max, mean, and median for a list of numeric scores."""
    if not scores:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
        }
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute confidence-score statistics for a terra-preta experiment."
    )
    parser.add_argument(
        "--exp-num",
        type=str,
        default="1",
        help="Experiment number (matches tmp/<exp-num>/ directory)",
    )
    args = parser.parse_args()
    exp_num = str(args.exp_num)

    experiment_dir = Path("tmp") / exp_num
    if not experiment_dir.exists():
        console.print(
            f"[red]Experiment directory '{experiment_dir}' does not exist.[/red]"
        )
        return

    # Load confirmed/control ids so we can categorise results.
    confirmed_data, control_data = get_confirmed_and_control_sites_for_terra_preta()
    confirmed_ids = {item["id"] for item in confirmed_data}
    control_ids = {item["id"] for item in control_data}

    all_scores: list[int] = []
    confirmed_scores: list[int] = []
    control_scores: list[int] = []

    # Iterate over each cell directory inside the experiment directory
    for cell_dir in experiment_dir.iterdir():
        if not cell_dir.is_dir():
            continue
        analysis_path = cell_dir / "analysis.json"
        if not analysis_path.exists():
            # Skip cells without completed analysis.
            continue
        try:
            with analysis_path.open("r") as f:
                analysis = json.load(f)
            score = int(analysis.get("confidence_score"))
        except Exception as e:
            console.print(
                f"[yellow]Warning: could not read confidence score from {analysis_path}: {e}[/yellow]"
            )
            continue

        cell_id = cell_dir.name
        all_scores.append(score)
        if cell_id in confirmed_ids:
            confirmed_scores.append(score)
        elif cell_id in control_ids:
            control_scores.append(score)
        else:
            # Unknown category – still counts for overall stats but warn user.
            console.print(
                f"[yellow]Warning: cell id '{cell_id}' not found in confirmed or control lists.[/yellow]"
            )

    # Prepare statistics tables
    overall_stats = _compute_stats(all_scores)
    confirmed_stats = _compute_stats(confirmed_scores)
    control_stats = _compute_stats(control_scores)

    table = Table(title=f"Experiment {exp_num} – Confidence-Score Summary")
    table.add_column("Category", style="bold cyan")
    table.add_column("Count", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")

    def _add_row(label: str, scores: list[int], stats: dict[str, float]):
        table.add_row(
            label,
            str(len(scores)),
            f"{stats['min']:.2f}",
            f"{stats['max']:.2f}",
            f"{stats['mean']:.2f}",
            f"{stats['median']:.2f}",
        )

    _add_row("All Cells", all_scores, overall_stats)
    _add_row("Confirmed", confirmed_scores, confirmed_stats)
    _add_row("Control", control_scores, control_stats)

    console.print(table)


if __name__ == "__main__":
    main()
