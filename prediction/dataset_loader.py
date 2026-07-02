import pandas as pd
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):

    def __init__(
        self,
        csv_file,
        seq_len=10,
        pred_len=5
    ):

        self.seq_len = seq_len
        self.pred_len = pred_len

        df = pd.read_csv(csv_file)

        self.samples = []

        for obj_id in df["object_id"].unique():

            obj_df = df[
                df["object_id"] == obj_id
            ]

            coords = obj_df[
                ["x_center", "y_center"]
            ].values

            if len(coords) < seq_len + pred_len:
                continue

            for i in range(
                len(coords)
                - seq_len
                - pred_len
                + 1
            ):

                # Normalize coordinates
                past = (
                    coords[
                        i:i + seq_len
                    ] / 416.0
                )

                future = (
                    coords[
                        i + seq_len:
                        i + seq_len + pred_len
                    ] / 416.0
                )

                self.samples.append(
                    (past, future)
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        past, future = self.samples[idx]

        return (
            torch.tensor(
                past,
                dtype=torch.float32
            ),
            torch.tensor(
                future,
                dtype=torch.float32
            )
        )