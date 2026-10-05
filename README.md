# Football ball tracker

This project detects and tracks a football in an MP4 video. It draws the ball and a pitch-position map on the output video, and writes per-frame ball coordinates to a CSV file. The processing logic comes from `main.ipynb`.

## Install with Conda

```bash
conda create -n balltracker python=3.11 -y
conda activate balltracker
pip install -r requirements.txt
```

## Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then run:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Set a Roboflow API key for the field detection model:

```bash
export ROBOFLOW_API_KEY="your_key_here"
```

Set a Hugging Face token for the ball detector weights:

```bash
export HF_TOKEN="your_key_here"
```

The ball detector weights are downloaded from Hugging Face on the first run. A CUDA GPU is recommended for inference.

## Run

```bash
python main.py path/to/video.mp4
python main.py path/to/video.mp4 --output-name match1
python main.py path/to/video.mp4 match1
```

The commands write `results/output.mp4` and `results/output.csv`, or `results/match1.mp4` and `results/match1.csv` when an output name is given. The video contains the ball annotation and pitch map; the CSV records image and pitch coordinates for each frame. Frame processing progress is shown in the terminal.
