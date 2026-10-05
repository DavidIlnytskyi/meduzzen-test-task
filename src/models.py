"""Load the two models used for ball and field detection."""

import os
import logging


FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"
BALL_MODEL_REPO = "julianzu9612/RFDETR-Soccernet"
BALL_MODEL_WEIGHTS = "weights/checkpoint_best_regular.pth"


def load_models():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Set ROBOFLOW_API_KEY before running the video pipeline.")

    # These imports are deferred so --help and input validation work without
    # downloading or initializing either model.
    import torch
    from huggingface_hub import hf_hub_download
    from inference import get_model
    from rfdetr import RFDETRLargeDeprecated

    # RF-DETR configures its own logger during import, so apply the level here.
    logging.getLogger("rf-detr").setLevel(logging.ERROR)

    field_model = get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=api_key)
    weights_path = hf_hub_download(repo_id=BALL_MODEL_REPO, filename=BALL_MODEL_WEIGHTS, token=os.environ["HF_TOKEN"],)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ball_model = RFDETRLargeDeprecated(
        pretrain_weights=str(weights_path),
        num_classes=3,
        device=device,
    )
    if device == "cuda":
        ball_model.inference(dtype=torch.float16)

    return ball_model, field_model
