import asyncio
import json
import random
from pathlib import Path

from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.config import settings
from src.evals.terra_preta.data import get_confirmed_and_control_sites_for_terra_preta
from src.evals.terra_preta.prompts import INSTRUCTIONS, TOOLS
from src.evals.terra_preta.tools import ExperimentManager
from src.utils import encode_image_pil

random.seed(41)
sem = asyncio.Semaphore(3)


async def generate(item: dict):
    async with sem:
        experiment_id = item["exp_num"]
        target_cell_id = item["id"]
        target_cell_polygon = item["geometry"]
        neighbors = item["neighbors"]
        site_type = item["site_type"]
        print(f"Processing {target_cell_id} {site_type}")
        history = []

        manager = ExperimentManager(
            experiment_id=experiment_id,
            target_cell_id=target_cell_id,
            target_cell_polygon=target_cell_polygon,
            neighbors=neighbors,
        )
        closest_river = manager.get_closest_river()
        mapbox_map = manager.get_mapbox_map()
        nearest_neighbors_map = manager.get_nearest_neighbors_map()
        all_imgs = (
            closest_river["images"]
            + mapbox_map["images"]
            + nearest_neighbors_map["images"]
        )
        content = []
        for img in all_imgs:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encode_image_pil(img)}",
                    "detail": "high",
                }
            )

        user_msg = f"""
        Here is the information about the cell:

        ID: {target_cell_id}
        Neighbor Cell IDs: {list(neighbors.keys())} (You can see their ids on the map too, they are labeled with their ids, use the name directly if you want vegetation index for them, example: `get_vegetation_indices(cell_id='northeast', ...)`

        Closest River Information: {closest_river["response"]}
        You have also been provided with a Mapbox satellite image of the cell.

        You may begin your analysis now."""

        content.append(
            {
                "type": "input_text",
                "text": user_msg,
            },
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        history.extend(messages)
        response = await settings.async_openai_client.responses.create(
            instructions=INSTRUCTIONS,
            input=messages,
            tools=TOOLS,
            model="o3-2025-04-16",
            reasoning={"summary": "auto", "effort": "medium"},
            max_output_tokens=20_000,
            store=True,
            parallel_tool_calls=False,
            metadata={
                "key": target_cell_id,
            },
            timeout=3600,
        )
        previous_response_id = response.id

        try:
            while True:
                messages = []
                previous_response_id = response.id
                for r in response.output:
                    history.append(r.model_dump())
                    if r.type == "function_call":
                        fn_output, user_reply = manager.process_tool_call(r)
                        if fn_output is None and user_reply is None:
                            # analysis finished
                            with open(
                                manager.target_cell_output_dir / "history.json", "w"
                            ) as f:
                                json.dump(history, f)
                            with open(
                                manager.target_cell_output_dir / "analysis.json", "w"
                            ) as f:
                                json.dump(json.loads(r.arguments), f)
                            total_tokens = response.usage.total_tokens
                            with open(
                                manager.target_cell_output_dir / "total_tokens.txt", "w"
                            ) as f:
                                f.write(str(total_tokens))
                            print(f"Finished {target_cell_id} {site_type}")
                            return target_cell_id, None
                        else:
                            messages.append(fn_output)
                            history.append(fn_output)
                            if user_reply:
                                messages.append(user_reply)
                                history.append(user_reply)

                if len(messages) == 0:
                    # no function calls, add a helper message
                    msg = {
                        "role": "user",
                        "content": "Please call a relevant tool to continue the analysis or if you are done, call the `finish_analysis` tool with your analysis.",
                    }
                    messages.append(msg)
                    history.append(msg)

                response = await settings.async_openai_client.responses.create(
                    input=messages,
                    previous_response_id=previous_response_id,
                    tools=TOOLS,
                    model="o3-2025-04-16",
                    reasoning={"summary": "auto", "effort": "medium"},
                    max_output_tokens=30_000,
                    store=True,
                    parallel_tool_calls=False,
                    metadata={
                        "key": target_cell_id,
                    },
                    timeout=3600,
                )
        except Exception as e:
            print(f"Error {target_cell_id}: {e}")
            return target_cell_id, str(e)


async def generate_stream(batch: list[dict]):
    """Yield generate() results as soon as each completes."""
    coros = [generate(item) for item in batch]
    for coro in asyncio.as_completed(coros):
        yield await coro


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-num", type=int, default=1, help="Experiment number")
    args = parser.parse_args()
    exp_num = str(args.exp_num)
    confirmed_data, control_data = get_confirmed_and_control_sites_for_terra_preta()

    # choose 10 random confirmed sites
    confirmed_sites = random.sample(confirmed_data, 50)
    control_sites = random.sample(control_data, 50)

    batch: list[dict] = []
    for confirmed_site in confirmed_sites:
        batch.append(
            {
                "id": confirmed_site["id"],
                "geometry": confirmed_site["geometry"],
                "neighbors": confirmed_site["neighbors"],
                "site_type": "confirmed",
                "exp_num": exp_num,
            }
        )
    for control_site in control_sites:
        batch.append(
            {
                "id": control_site["id"],
                "geometry": control_site["geometry"],
                "neighbors": control_site["neighbors"],
                "site_type": "control",
                "exp_num": exp_num,
            }
        )

    errors = {}
    with Progress(
        SpinnerColumn(),
        "[bold blue]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("Processing tiles", total=len(batch))

        async for id, error in generate_stream(batch):
            if error:
                errors[id] = error
            progress.update(task_id, advance=1)

    errors_path = Path(f"tmp/errors_{exp_num}.json")
    with errors_path.open("w") as f:
        json.dump(errors, f)


if __name__ == "__main__":
    asyncio.run(main())
