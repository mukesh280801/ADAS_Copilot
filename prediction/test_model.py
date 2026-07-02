import torch
from transformer_model import TrajectoryTransformer
from dataset_loader import TrajectoryDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TrajectoryDataset(
    "prediction/trajectory_dataset.csv"
)

model = TrajectoryTransformer().to(device)

model.load_state_dict(
    torch.load(
        "prediction/trajectory_transformer.pth",
        map_location=device
    )
)

model.eval()

past, future = dataset[0]

past = past.unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(past)

print("\nGround Truth:")
print(future)

print("\nPrediction:")
print(pred.squeeze(0).cpu())