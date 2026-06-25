from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path


class Table:
    """Small CSV table helper used to avoid heavyweight plotting dependencies."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    @property
    def columns(self) -> set[str]:
        cols: set[str] = set()
        for row in self.rows:
            cols.update(row.keys())
        return cols

    @property
    def empty(self) -> bool:
        return len(self.rows) == 0

    def copy(self) -> "Table":
        return Table([dict(row) for row in self.rows])

    def filter_equal(self, column: str, value) -> "Table":
        return Table([row for row in self.rows if row.get(column) == value])

    def mode(self, column: str):
        values = [row.get(column) for row in self.rows if column in row]
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]

    def values(self, column: str) -> list:
        return [row.get(column) for row in self.rows]

    def to_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted(self.columns)
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        replace_with_retry(tmp_path, path)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


def read_csv(path: Path) -> Table:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return Table([{key: parse_scalar(value) for key, value in row.items()} for row in reader])


def write_csv(path: Path, rows: list[dict]) -> None:
    Table(rows).to_csv(path)


def replace_with_retry(src: Path, dst: Path, attempts: int = 20, delay: float = 0.1) -> None:
    for attempt in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def parse_scalar(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return ""
    if text == "False":
        return False
    if text == "True":
        return True
    if text.startswith("[") and text.endswith("]"):
        return text
    try:
        if any(token in text for token in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def sorted_unique(values: list) -> list:
    cleaned = [value for value in values if value is not None]
    try:
        return sorted(set(cleaned))
    except TypeError:
        seen = []
        for value in cleaned:
            if value not in seen:
                seen.append(value)
        return seen
