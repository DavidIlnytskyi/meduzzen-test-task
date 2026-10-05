"""Run ball tracking on an MP4 video."""

import argparse
from pathlib import Path
import os
import warnings
import logging
from collections.abc import Sequence

os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("inference").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


def output_stem(value: str) -> str:
    """Validate and normalize an output video file name."""
    name = Path(value).name
    if name != value or "\\" in value or name in {"", ".", ".."}:
        raise argparse.ArgumentTypeError("output name must be a file name, not a path")
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    if not name:
        raise argparse.ArgumentTypeError("output name cannot be empty")
    return name


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the video tracking pipeline."""
    parser = argparse.ArgumentParser(description="Track a football and draw its pitch position.")
    parser.add_argument("video", type=Path, help="path to an input MP4 video")
    parser.add_argument(
        "positional_output_name", nargs="?", type=output_stem,
        help="optional output file name (default: output)",
    )
    parser.add_argument(
        "--output-name", "-o", type=output_stem,
        help="output file name in results/",
    )
    args = parser.parse_args(argv)

    if args.output_name and args.positional_output_name:
        parser.error("provide the output name either positionally or with --output-name")
    name = args.output_name or args.positional_output_name or "output"

    if args.video.suffix.lower() != ".mp4" or not args.video.is_file():
        parser.error("video must be an existing .mp4 file")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    video_path = results_dir / f"{name}.mp4"
    csv_path = results_dir / f"{name}.csv"
    if args.video.resolve() == video_path.resolve():
        parser.error("input video and output video cannot be the same file")

    print("Execution start...", flush=True)
    try:
        from src.pipeline import process_video

        process_video(args.video, video_path, csv_path)
    except ModuleNotFoundError as exc:
        parser.exit(1, f"error: missing package {exc.name}; run pip install -r requirements.txt\n")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Video saved to {video_path}")
    print(f"Ball positions saved to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
