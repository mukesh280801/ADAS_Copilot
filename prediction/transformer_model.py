import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):

    def __init__(
        self,
        input_dim=2,
        d_model=128,
        nhead=4,
        num_layers=4,
        pred_len=5
    ):

        super().__init__()

        self.embedding = nn.Linear(
            input_dim,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(
            d_model,
            pred_len * 2
        )

        self.pred_len = pred_len

    def forward(self, x):

        x = self.embedding(x)

        x = self.transformer(x)

        x = x[:, -1, :]

        x = self.fc(x)

        return x.view(
            -1,
            self.pred_len,
            2
        )