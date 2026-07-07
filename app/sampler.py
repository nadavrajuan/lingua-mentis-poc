from __future__ import annotations
import random
import sqlite3
from app.data import get_test_dataset

_test_indices_by_class: dict[int, list[int]] | None = None


def _class_index() -> dict[int, list[int]]:
    global _test_indices_by_class
    if _test_indices_by_class is None:
        ds = get_test_dataset()
        idx: dict[int, list[int]] = {i: [] for i in range(10)}
        for i, (_, label) in enumerate(ds):
            idx[int(label)].append(i)
        _test_indices_by_class = idx
    return _test_indices_by_class


def sample_image(mode: str, class_a: int | None, class_b: int | None, ambiguity_db: str) -> int:
    idx = _class_index()
    total = sum(len(v) for v in idx.values())

    if mode == "random":
        return random.randint(0, total - 1)
    elif mode == "only_a":
        c = class_a if class_a is not None else 3
        return random.choice(idx[c])
    elif mode == "only_b":
        c = class_b if class_b is not None else 8
        return random.choice(idx[c])
    elif mode in ("3_vs_8_ambiguous", "ambiguous", "low_confidence", "top2_confusion"):
        return _sample_from_bank(ambiguity_db, class_a, class_b, mode)
    elif mode == "custom_pair":
        pool = []
        if class_a is not None:
            pool.extend(idx[class_a])
        if class_b is not None:
            pool.extend(idx[class_b])
        if not pool:
            return random.randint(0, total - 1)
        return random.choice(pool)
    else:
        return random.randint(0, total - 1)


def _sample_from_bank(db_path: str, class_a: int | None, class_b: int | None, mode: str) -> int:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        if mode == "low_confidence":
            cur.execute("SELECT image_id FROM ambiguity WHERE confidence < 0.75 ORDER BY RANDOM() LIMIT 1")
        elif mode == "3_vs_8_ambiguous" and class_a is not None and class_b is not None:
            cur.execute(
                "SELECT image_id FROM ambiguity WHERE (true_label=? OR true_label=?) AND margin < 0.25 ORDER BY RANDOM() LIMIT 1",
                (class_a, class_b),
            )
        elif mode == "top2_confusion":
            cur.execute("SELECT image_id FROM ambiguity WHERE margin < 0.20 ORDER BY RANDOM() LIMIT 1")
        else:
            cur.execute("SELECT image_id FROM ambiguity ORDER BY RANDOM() LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    ds = get_test_dataset()
    return random.randint(0, len(ds) - 1)
