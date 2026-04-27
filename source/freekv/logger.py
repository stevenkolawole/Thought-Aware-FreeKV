"""Lightweight CSV-gz + JSON logger for FreeKV instrumentation runs.

Lazy-open: zero I/O when log_dir is None. Writers are thread-safe so the
non-blocking recall path (which can be touched from multiple CUDA streams)
won't corrupt rows.

CSV files use gzip; pandas reads them via pd.read_csv("foo.csv.gz") natively.
"""
import csv
import gzip
import json
import os
import threading
from typing import Any, Dict, List, Optional


class _CsvGzWriter:
    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        self._fp = None
        self._writer = None
        self._lock = threading.Lock()

    def write(self, row: Dict[str, Any]) -> None:
        with self._lock:
            if self._fp is None:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                self._fp = gzip.open(self.path, "wt", newline="")
                self._writer = csv.DictWriter(self._fp, fieldnames=self.fieldnames)
                self._writer.writeheader()
            self._writer.writerow(row)

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None


class RunLogger:
    DECODE_FIELDS = [
        "prompt_id", "step_id", "layer_id", "q_len",
        "cos_sim", "correction_triggered", "num_corr_heads",
        "ema_sim", "thought_type", "segment_id",
    ]
    RECALL_FIELDS = [
        "prompt_id", "step_id", "layer_id", "recall_kind",
        "recall_num_pages", "recall_bytes",
    ]

    def __init__(self, log_dir: Optional[str], run_tag: str = ""):
        self.log_dir = log_dir
        self.run_tag = run_tag
        self.decode = None
        self.recall = None
        if log_dir is not None:
            self.decode = _CsvGzWriter(
                os.path.join(log_dir, "decode_log.csv.gz"), self.DECODE_FIELDS
            )
            self.recall = _CsvGzWriter(
                os.path.join(log_dir, "recall_log.csv.gz"), self.RECALL_FIELDS
            )

    @property
    def enabled(self) -> bool:
        return self.log_dir is not None

    def write_meta(self, meta: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def append_gen(self, entry: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "gen_log.jsonl"), "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def close(self) -> None:
        if self.decode is not None:
            self.decode.close()
        if self.recall is not None:
            self.recall.close()
