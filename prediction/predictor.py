import torch
import numpy as np

from prediction.transformer_model import (
    TrajectoryTransformer
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = TrajectoryTransformer().to(device)

model.load_state_dict(
    torch.load(
        "prediction/trajectory_transformer.pth",
        map_location=device
    )
)

model.eval()


def predict_future(trajectory):

    if len(trajectory) < 10:
        return None

    trajectory = trajectory[-10:]

    trajectory = np.array(
        trajectory,
        dtype=np.float32
    )

    x = torch.tensor(
        trajectory / 416.0,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        pred = model(x)

    pred = pred.cpu().numpy()[0]

    pred = pred * 416.0

    return pred