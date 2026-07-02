import torch
from torch.utils.data import DataLoader

from dataset_loader import TrajectoryDataset
from transformer_model import TrajectoryTransformer

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

dataset = TrajectoryDataset(
    "prediction/trajectory_dataset.csv"
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

model = TrajectoryTransformer().to(device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 100

for epoch in range(EPOCHS):

    total_loss = 0

    for past, future in loader:

        past = past.to(device)
        future = future.to(device)

        pred = model(past)

        loss = criterion(
            pred,
            future
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" Loss={total_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "prediction/trajectory_transformer.pth"
)

print("✅ Model Saved")