import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Datapoint:
    X: np.ndarray
    y: float


def load_electricity_dataset(path: str | Path, head: int | None = None) -> np.ndarray:

    if isinstance(path, str):
        path = Path(path)

    if path.suffix.lower() == ".csv":
        csv_data = _read_csv(path, head=head)

    processed_data = [_process_data(row) for row in csv_data.get("data")]

    data = [
        Datapoint(X=np.array(row[:-1]), y=row[-1])
        for row in processed_data
        if len(row) >= 2
    ]

    return np.array(data)


def _process_data(x_values: list) -> list[int]:
    cleaned = []
    for val in x_values:
        if isinstance(val, str) and "b" in val.lower():
            val = val[2:-1]
        try:
            val = float(val)
        except ValueError:
            val = 1 if "UP" in val.upper() else 0

        cleaned.append(float(val))

    return cleaned


def _read_csv(path: Path, head: int | None = None):
    if not isinstance(head, int):
        head = 0
    with open(path, mode="r") as f:
        data = list(csv.reader(f))
        return {"column_names": data[0], "data": data[1 : head + 1 or -1]}
