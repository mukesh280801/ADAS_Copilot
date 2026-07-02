import csv
import os
from datetime import datetime

FILE_NAME = "risk_events.csv"

def log_risk(
    object_id,
    risk_score,
    min_distance
):

    file_exists = os.path.exists(
        FILE_NAME
    )

    with open(
        FILE_NAME,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "timestamp",
                "object_id",
                "risk_score",
                "min_distance"
            ])

        writer.writerow([
            datetime.now().strftime(
                "%H:%M:%S"
            ),
            object_id,
            round(risk_score, 4),
            round(min_distance, 4)
        ])