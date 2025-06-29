import asyncio
import json
from pathlib import Path

import fire
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.config import settings
from src.evals.lidar.prompts import INSTRUCTIONS, TOOLS

sem = asyncio.Semaphore(128)


async def generate(item: dict):
    # throttle concurrent API calls with the global semaphore
    async with sem:
        instructions = item["instructions"]
        messages = item["input"]
        tools = item["tools"]
        key = item["key"]

        try:
            response = await settings.async_openai_client.responses.create(
                instructions=instructions,
                input=messages,
                tools=tools,
                model="o3-2025-04-16",
                reasoning={"summary": "auto", "effort": "medium"},
                max_output_tokens=20_000,
                store=True,
                parallel_tool_calls=False,
                metadata={
                    "key": key,
                },
                timeout=3600,
            )
            print(f"Finished {key}")
            return (
                response.model_dump(),
                None,
                key,
            )
        except Exception as e:
            print(f"Error {key}: {e}")
            return None, str(e), key


async def generate_stream(batch: list[dict]):
    """Yield generate() results as soon as each completes."""
    coros = [generate(item) for item in batch]
    for coro in asyncio.as_completed(coros):
        yield await coro


async def main(exp_num: int, exp_type: str = "eval"):
    with open("data/evals/lidar_tiles_signed_urls.json", "r") as f:
        tiles = json.load(f)

    if exp_type == "eval":
        tiles = [t for t in tiles if t["subset"] != "test"]
    elif exp_type == "test":
        # For the dedicated test run we want to process only the tiles that were
        # explicitly tagged as the hold-out "test" subset in the metadata.
        tiles = [t for t in tiles if t["subset"] == "test"]
    else:
        raise ValueError("exp_type must be either 'eval' or 'test'")

    batch = []
    for t in tiles:
        user_msg = {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": t["zscore_url"],
                    "detail": "high",
                },
                {
                    "type": "input_image",
                    "image_url": t["dem_url"],
                    "detail": "high",
                },
                {
                    "type": "input_text",
                    "text": "Here are the images you need to analyze. Good luck!",
                },
            ],
        }

        batch.append(
            {
                "input": [user_msg],
                "tools": TOOLS,
                "instructions": INSTRUCTIONS,
                "key": t["tile_key"],
            }
        )

    if not batch:
        print("No new tiles to process – exiting early.")
        return

    print(f"Processing {len(batch)} tiles...")
    results_path = Path(f"data/evals/lidar_eval_output_{exp_type}_{exp_num}.json")
    results = {}
    with Progress(
        SpinnerColumn(),
        "[bold blue]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("Processing tiles", total=len(batch))

        async for output, error, key in generate_stream(batch):
            results[key] = {
                "output": output,
                "error": error,
            }
            progress.update(task_id, advance=1)

    with results_path.open("w") as f:
        json.dump(results, f)


# ---- CLI entry -------------------------------------------------------------


def _cli(exp_num: int, exp_type: str = "eval") -> None:
    """Command-line wrapper that executes the async main() inside asyncio.run()."""

    # Fire passes arguments as str by default; ensure correct types
    exp_num_int = int(exp_num)
    asyncio.run(main(exp_num_int, exp_type))


if __name__ == "__main__":
    # Let Fire parse the command-line and call our synchronous wrapper.
    fire.Fire(_cli)
