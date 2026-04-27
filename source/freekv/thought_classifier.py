"""Trajectory-based thought-type classifier.

Consumes per-step cos-sim values (FreeKV already computes these for the
correction check) and produces R/E/T segment labels.

Two cadences:
  - Per-step EMA update: cheap, single-scalar state.
  - Per-window threshold check: only re-evaluates the segment label every
    `segment_window` decode steps to avoid label jitter. Segment ID
    increments only when the label actually crosses a threshold.

Thresholds and EMA alpha are placeholders; sweep on calibration data.

Same code is callable both:
  (a) from the runtime (InferState owns one instance), and
  (b) offline against a logged decode_log.csv.gz (one classifier per prompt,
      replay cos_sim values in step order). See scripts/backfill_thought_labels.py.
"""
from .utils import ThoughtType


class ThoughtClassifier:
    def __init__(
        self,
        ema_alpha: float = 0.1,
        r_threshold: float = 0.92,
        t_threshold: float = 0.75,
        segment_window: int = 16,
    ):
        self.ema_alpha = ema_alpha
        self.r_threshold = r_threshold
        self.t_threshold = t_threshold
        self.segment_window = max(1, int(segment_window))

        self.ema_sim: float | None = None
        self.current_type: ThoughtType = ThoughtType.R
        self.prev_type: ThoughtType = ThoughtType.R
        self.segment_id: int = 0
        self._last_check_step: int = -10**9

    def reset(self) -> None:
        self.ema_sim = None
        self.current_type = ThoughtType.R
        self.prev_type = ThoughtType.R
        self.segment_id = 0
        self._last_check_step = -10**9

    def update(self, cos_sim: float | None, step_id: int) -> None:
        if cos_sim is None or cos_sim != cos_sim:  # NaN guard
            return
        if self.ema_sim is None:
            self.ema_sim = cos_sim
        else:
            self.ema_sim = (
                self.ema_alpha * cos_sim
                + (1.0 - self.ema_alpha) * self.ema_sim
            )

        if step_id - self._last_check_step >= self.segment_window:
            new_type = self._classify()
            if new_type != self.current_type:
                self.prev_type = self.current_type
                self.current_type = new_type
                self.segment_id += 1
            self._last_check_step = step_id

    def _classify(self) -> ThoughtType:
        if self.ema_sim is None:
            return ThoughtType.R
        if self.ema_sim >= self.r_threshold:
            return ThoughtType.R
        if self.ema_sim <= self.t_threshold:
            return ThoughtType.T
        return ThoughtType.E

    @property
    def in_transition(self) -> bool:
        return self.current_type == ThoughtType.T

    @property
    def just_left_transition(self) -> bool:
        return (
            self.prev_type == ThoughtType.T
            and self.current_type != ThoughtType.T
        )
