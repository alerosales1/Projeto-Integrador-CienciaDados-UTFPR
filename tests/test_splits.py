import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from splits import forward_chaining_folds  # noqa: E402

FAILURES = [
    {"id": 1, "start": datetime(2020, 4, 18, 0, 0), "end": datetime(2020, 4, 18, 23, 59)},
    {"id": 2, "start": datetime(2020, 5, 29, 23, 30), "end": datetime(2020, 5, 30, 6, 0)},
    {"id": 3, "start": datetime(2020, 6, 5, 10, 0), "end": datetime(2020, 6, 7, 14, 30)},
    {"id": 4, "start": datetime(2020, 7, 15, 14, 30), "end": datetime(2020, 7, 15, 19, 0)},
]


def test_fold_count_and_order():
    folds = forward_chaining_folds(FAILURES, embargo_hours=24)
    assert [f.test_failure_id for f in folds] == [2, 3, 4]


def test_train_precedes_test_with_embargo():
    for f in forward_chaining_folds(FAILURES, embargo_hours=24):
        assert f.train_end < f.test_start < f.test_end
        assert (f.test_start - f.train_end).total_seconds() == 24 * 3600


def test_train_never_contains_the_tested_failure():
    by_id = {f["id"]: f for f in FAILURES}
    for f in forward_chaining_folds(FAILURES, embargo_hours=24):
        assert f.train_end < by_id[f.test_failure_id]["start"]


def test_input_order_does_not_matter():
    a = forward_chaining_folds(FAILURES, 24)
    b = forward_chaining_folds(list(reversed(FAILURES)), 24)
    assert a == b
