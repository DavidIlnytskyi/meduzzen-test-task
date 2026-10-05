# Football ball tracker

This project detects and tracks a football in MP4 match footage. It produces an annotated video with a pitch-position map and a CSV of per-frame image and pitch coordinates. The pipeline is implemented in `main.py` and `src/`.

## Videos

The original clips are in `data/`. Their processed videos and tracking coordinates are in `results/`.

## Example result

### Crop 1 — Validation

<table>
<tr>
<td align="center"><strong>Input</strong></td>
<td align="center"><strong>Processed</strong></td>
</tr>
<tr>
<td>

<video controls preload="metadata" width="440" src="data/crop1.mp4"></video>

</td>
<td>

<video controls preload="metadata" width="440" src="results/crop1.mp4"></video>

</td>
</tr>
<tr>
<td align="center">

[Open input video](data/crop1.mp4)

</td>
<td align="center">

[Open processed video](results/crop1.mp4) · [Coordinates CSV](results/crop1.csv)

</td>
</tr>
</table>

---

### Crop 2 — Test


<table>
<tr>
<td align="center"><strong>Input</strong></td>
<td align="center"><strong>Processed</strong></td>
</tr>
<tr>
<td>

<video controls preload="metadata" width="440" src="data/crop2.mp4"></video>

</td>
<td>

<video controls preload="metadata" width="440" src="results/crop2.mp4"></video>

</td>
</tr>
<tr>
<td align="center">

[Open input video](data/crop2.mp4)

</td>
<td align="center">

[Open processed video](results/crop2.mp4) · [Coordinates CSV](results/crop2.csv)

</td>
</tr>
</table>


## Data Analysis

The data consists of full-length recordings of football matches. An important characteristic of this data is that the videos contain many fouls, throw-ins, and rapid ball movements from one part of the field to another. Another important issue is the presence of players whose heads or other body parts can visually resemble the ball, causing the detector to produce false positives. *(TODO: provide a short explanation of the main types of false positives.)*

For this reason, it is important to select appropriate video fragments for testing. Based on the characteristics described above, the goal was to select two 30–60 second clips where the ball remains in play and there are no fouls or throw-ins. At the same time, the selected clips should contain ball occlusions and long passes or shots from one part of the field to another so that they remain representative of the overall data.

The videos were cropped using FFmpeg.

An important limitation is that no ground-truth labels were provided for this task, which prevents direct evaluation using tracking metrics such as MOTA, IDF1, and HOTA. Although several approaches could be used to obtain labels, including manual annotation, automatic labeling using models such as ChatGPT Astra or Claude Opus, or evaluation on an existing labeled dataset, the limited time available for this test task made empirical evaluation the primary approach.

The two fragments are `crop1.mp4` (validation) and `crop2.mp4` (test).

## Approach Evolution

### Model Selection

Because of the limited amount of time available for the task, a pretrained detector was used. Two main candidates were considered:

- [YOLO football ball detector](https://huggingface.co/martinjolif/yolo-football-ball-detection)
- [RF-DETR SoccerNet](https://huggingface.co/julianzu9612/RFDETR-Soccernet)

RF-DETR was selected because it provides better metrics for the specific ball-detection class. In general, RF-DETR requires more computational resources and longer training time, while YOLO is more lightweight and typically faster. However, because detection accuracy was more important for this task, RF-DETR was selected.

### 2D Mapping

For football-pitch keypoint detection and subsequent 2D mapping, the following pretrained model was used as the base:

[Football Field Detection — Roboflow](https://huggingface.co/Simon9/football-field-detection-roboflow)

### 1. Baseline — RF-DETR Detector

The first approach used the RF-DETR detector without any tracking component in order to better understand the problem. This baseline produced many false positives and frequent target switching. Because of this, the next step was to introduce a Kalman filter to improve motion prediction and reduce the impact of false positives such as players' heads, legs, arms, and spectators.

### 2. RF-DETR + Kalman Filter

Adding the Kalman filter improved ball tracking. However, false positives still had a significant impact on tracking behavior, so further improvements were required.

### 3. RF-DETR + ByteTrack

ByteTrack was considered a stronger alternative because it is a ready-to-use tracking algorithm that already incorporates Kalman-filter-based motion prediction. One limitation is that its motion model works best when object movement is relatively predictable, while a football can move highly nonlinearly.

Although this approach improved tracking, it still produced many false positives and target switches. This indicated that improving the detector itself was also necessary.

### 4. RF-DETR + OC-SORT

To better handle nonlinear motion, OC-SORT was evaluated as the main tracking algorithm. It improved tracking behavior, particularly during changes in ball direction, but occasional false detections still remained.

### 5. RF-DETR + OC-SORT + Tiling

To reduce false positives and improve small-ball detection, several possible directions were considered:

- fine-tune the detector;
- use a different detector;
- perform inference on tiled video frames.

Tiled inference effectively increases the relative size of the ball seen by the detector. This makes it easier for the model to recognize small balls and reduces the number of false positives.

### 6. RF-DETR + Tiling + Custom Tracker

Tiling reduced the false-positive rate, but tracking problems still remained. Existing trackers treat detections across the entire frame relatively equally, while in this use case the ball has strong spatial dependency: with high probability, its next position will be located relatively close to or in the motion direction of its previous position.

Another limitation is that OC-SORT is primarily designed for multi-object tracking, while this task requires tracking only a single object. For this reason, a custom single-object tracker was implemented.

In the final pipeline, RF-DETR detects the ball in overlapping frame tiles, and non-maximum suppression removes duplicate detections. The custom tracker predicts the ball's next image position from its smoothed velocity, then chooses the closest detection within a search radius that grows after missed frames. It confirms a new or recovered track only after nearby detections appear in consecutive frames; after a longer loss, a sufficiently confident detection elsewhere in the frame can begin recovery. Field keypoints provide a homography for mapping confirmed ball positions onto the pitch, and the video and CSV record the result when that mapping is valid.

### Limitations / Improvements

1. Ball mapping onto the 2D pitch is inaccurate during shots or passes that significantly change the ball's height. A simple homography assumes that the object lies on the pitch plane and therefore cannot correctly represent vertical ball movement.

2. Camera calibration could be performed to estimate intrinsic camera parameters and compensate for lens distortion.

3. In the current implementation, ball detection is performed over the entire video frame. This creates problems because objects outside the pitch, including spectators, may be incorrectly classified as the ball. A possible improvement would be to reject detections outside the detected pitch boundaries. However, this introduces another problem for long shots where the ball may temporarily appear outside the projected pitch area in the image, so additional research is required.

4. The PnL package was not used because it requires a large number of dependencies (50+) and significant setup and build time. Instead, a Roboflow-based keypoint extraction model was used for football-pitch keypoint detection.

5. The current detector produces false positives by detecting players' heads, legs, and arms as the ball. For this task, the detector was obtained from an open-source source as an easy-to-use solution. A better long-term approach would be to fine-tune or train a dedicated ball detector. Training could include hard-negative examples to reduce false positives, together with tiling, small-ball augmentation, and occlusion augmentation to improve ball detection.

6. Another possible improvement is to assign a lower probability to ball detections located inside player bounding boxes. This would require an additional player detector but could reduce false positives and improve overall tracking accuracy.

7. The current pipeline assumes relatively clean input where the video continuously represents active gameplay. In the future, the pipeline should also handle camera changes, replays, highlights, and other broadcast transitions. A small classification model could be trained to determine the current scene type and route each frame through the appropriate pipeline branch.

## Setup

### Install with Conda

```bash
conda create -n balltracker python=3.11 -y
conda activate balltracker
pip install -r requirements.txt
```

### Install with uv

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

### Run

```bash
python main.py data/crop1.mp4 --output-name crop1
python main.py data/crop2.mp4 --output-name crop2
```

The first command writes `results/crop1.mp4` and `results/crop1.csv`; the second writes `results/crop2.mp4` and `results/crop2.csv`. Each video contains the ball annotation and pitch map, and each CSV records image and pitch coordinates for every frame. Frame processing progress is shown in the terminal. Omit `--output-name` to write `results/output.mp4` and `results/output.csv`.
