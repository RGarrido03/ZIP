"""Unit tests for models/queue_wait.py — queue waiting-time estimation."""

import time
from collections import deque

import numpy as np
import pytest
import torch

from models.queue_wait import (
    AdaptiveQueueROI,
    QueueWaitEstimator,
    SumPool2d,
    estimate_waiting_time,
    extract_irregular_queue,
    extract_queue_region,
)


# ---------------------------------------------------------------------------
# SumPool2d
# ---------------------------------------------------------------------------


class TestSumPool2d:
    def test_basic_sum(self):
        sp = SumPool2d(kernel_size=4, stride=2)
        x = torch.ones(1, 1, 8, 8)
        out = sp(x)
        assert out.shape == (1, 1, 3, 3)
        # Each 4×4 window → 16 ones → sum = 16
        assert torch.allclose(out, torch.full_like(out, 16.0))

    def test_non_unit_input(self):
        sp = SumPool2d(kernel_size=2, stride=2)
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        out = sp(x)
        assert out.item() == 10.0  # 1+2+3+4

    def test_stride_equals_kernel(self):
        sp = SumPool2d(kernel_size=3, stride=3)
        x = torch.eye(6, 6).unsqueeze(0).unsqueeze(0)
        out = sp(x)
        # 6×6 / 3×3 = 2×2 output grid; each window has 3 non-overlapping ones
        assert out.shape == (1, 1, 2, 2)


# ---------------------------------------------------------------------------
# extract_queue_region
# ---------------------------------------------------------------------------


class TestExtractQueueRegion:
    def test_single_blob(self):
        dmap = torch.zeros(64, 64)
        dmap[30:50, 5:35] = 1.0  # 20×30 = 600 px, sum = 600
        count, mask = extract_queue_region(dmap, percentile=50.0, smooth_kernel=7)
        assert count > 0
        assert mask.sum() > 100

    def test_empty_map(self):
        dmap = torch.zeros(64, 64)
        count, mask = extract_queue_region(dmap)
        assert count == 0.0
        assert mask.sum() == 0

    def test_batched_input_1_1_H_W(self):
        dmap = torch.zeros(1, 1, 64, 64)
        dmap[0, 0, 30:50, 5:35] = 1.0
        count, mask = extract_queue_region(dmap)
        assert count > 0
        assert mask.shape == (64, 64)

    def test_batched_input_1_H_W(self):
        dmap = torch.zeros(1, 64, 64)
        dmap[0, 30:50, 5:35] = 1.0
        count, mask = extract_queue_region(dmap)
        assert count > 0

    def test_two_blobs_takes_largest(self):
        dmap = torch.zeros(64, 64)
        dmap[10:15, 10:15] = 1.0  # small blob: 25 px
        dmap[30:50, 5:35] = 1.0  # large blob: 600 px
        count, mask = extract_queue_region(dmap, percentile=50.0, smooth_kernel=3)
        # The largest blob should be selected
        assert mask[30:50, 5:35].sum() > mask[10:15, 10:15].sum()


# ---------------------------------------------------------------------------
# extract_irregular_queue
# ---------------------------------------------------------------------------


class TestExtractIrregularQueue:
    def test_basic(self):
        dmap = torch.zeros(64, 64)
        dmap[30:50, 5:35] = 1.0
        count = extract_irregular_queue(dmap, min_density=0.05)
        assert count > 0

    def test_empty(self):
        dmap = torch.zeros(64, 64)
        count = extract_irregular_queue(dmap)
        assert count == 0.0

    def test_serpentine_queue(self):
        """Simulate a winding queue: several disconnected blobs that
        dilation should connect."""
        dmap = torch.zeros(64, 64)
        # Three nearby blobs forming a snake
        dmap[10:15, 10:15] = 1.0
        dmap[14:20, 20:25] = 1.0
        dmap[19:24, 30:35] = 1.0
        count = extract_irregular_queue(
            dmap, min_density=0.05, dilate_iters=10, open_iters=2
        )
        # Dilation should connect all three → one blob
        assert count > 0


# ---------------------------------------------------------------------------
# AdaptiveQueueROI
# ---------------------------------------------------------------------------


class TestAdaptiveQueueROI:
    def test_convergence(self):
        dmap = torch.zeros(64, 64)
        dmap[30:50, 5:35] = 1.0
        roi = AdaptiveQueueROI(alpha=0.1, quantile=0.5)
        c1 = roi.update(dmap)
        assert c1 > 0
        # Feed the same frame multiple times — should remain stable
        for _ in range(20):
            c = roi.update(dmap)
        assert c > 0

    def test_reset(self):
        dmap = torch.ones(64, 64) * 0.5
        roi = AdaptiveQueueROI()
        roi.update(dmap)
        assert roi.running_mean is not None
        roi.reset()
        assert roi.running_mean is None

    def test_empty_then_nonempty(self):
        roi = AdaptiveQueueROI()
        c = roi.update(torch.zeros(64, 64))
        assert c == 0.0
        # Now feed a non-empty map — ROI should learn it
        dmap = torch.zeros(64, 64)
        dmap[20:40, 20:40] = 1.0
        for _ in range(10):
            c = roi.update(dmap)
        assert c > 0

    def test_alpha_bounds(self):
        with pytest.raises(ValueError):
            AdaptiveQueueROI(alpha=0)
        with pytest.raises(ValueError):
            AdaptiveQueueROI(alpha=-0.1)
        with pytest.raises(ValueError):
            AdaptiveQueueROI(alpha=1.5)
        # alpha=1.0 is valid
        AdaptiveQueueROI(alpha=1.0)

    def test_uniform_density(self):
        """All density values equal → threshold should still produce a mask."""
        dmap = torch.ones(64, 64)
        roi = AdaptiveQueueROI(alpha=0.5, quantile=0.3)
        c = roi.update(dmap)
        assert c > 0


# ---------------------------------------------------------------------------
# QueueWaitEstimator
# ---------------------------------------------------------------------------


class TestQueueWaitEstimator:
    @staticmethod
    def _make_dmap(shape=(64, 64), density=0.0, region=None):
        dmap = torch.zeros(shape)
        if region is not None:
            y1, y2, x1, x2 = region
            dmap[y1:y2, x1:x2] = density
        else:
            dmap.fill_(density)
        return dmap

    # --- construction ---

    def test_default_strategy_is_ema(self):
        est = QueueWaitEstimator()
        assert est.strategy == "ema"
        assert est._roi is not None

    def test_threshold_strategy_has_no_roi(self):
        est = QueueWaitEstimator(strategy="threshold")
        assert est._roi is None

    def test_invalid_fps(self):
        with pytest.raises(ValueError):
            QueueWaitEstimator(fps=0)
        with pytest.raises(ValueError):
            QueueWaitEstimator(fps=-1)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            QueueWaitEstimator(window_seconds=0)

    # --- first frame returns None ---

    def test_first_frame_returns_none(self):
        est = QueueWaitEstimator(strategy="ema")
        dmap = self._make_dmap(region=(30, 50, 5, 35), density=1.0)
        w = est.update(dmap)
        assert w is None

    # --- draining queue: reasonable wait ---

    def test_draining_queue_reasonable_wait(self):
        est = QueueWaitEstimator(strategy="threshold", fps=10, window_seconds=10)
        # Simulate a draining queue by manually injecting history
        now = time.time()
        est._history = deque(
            [
                (now - 10.0, 100.0),  # 10s ago, 100 people
                (now - 8.0, 90.0),
                (now - 6.0, 80.0),
                (now - 4.0, 70.0),
                (now - 2.0, 60.0),
                (now, 50.0),  # now, 50 people
            ],
            maxlen=100,
        )
        w = est._estimate_wait(50.0)
        # Delta = 100 - 50 = 50 people in 10s → rate = 5 ppl/s
        # Wait = 50 / 5 = 10s
        assert w is not None
        assert 8.0 <= w <= 12.0  # allow float tolerance

    # --- stable queue: floor applies ---

    def test_stable_queue_uses_floor(self):
        est = QueueWaitEstimator(strategy="ema", fps=10, window_seconds=10)
        now = time.time()
        est._history = deque(
            [
                (now - 10.0, 50.0),
                (now, 50.0),
            ],
            maxlen=100,
        )
        w = est._estimate_wait(50.0)
        assert w is not None
        # Delta = 0 → rate floors at 0.05 → wait = 50 / 0.05 = 1000s
        assert w == pytest.approx(1000.0, rel=0.01)

    # --- growing queue: floor applies (conservative) ---

    def test_growing_queue_uses_floor(self):
        est = QueueWaitEstimator(strategy="ema", fps=10, window_seconds=10)
        now = time.time()
        est._history = deque(
            [
                (now - 5.0, 10.0),
                (now, 50.0),
            ],
            maxlen=100,
        )
        w = est._estimate_wait(50.0)
        assert w is not None
        # Delta = 10-50 = -40, rate floors at 0.05, wait = 50/0.05 = 1000
        assert w == pytest.approx(1000.0, rel=0.01)

    # --- empty queue ---

    def test_empty_queue_zero_wait(self):
        est = QueueWaitEstimator(strategy="threshold", fps=10, window_seconds=10)
        now = time.time()
        est._history = deque(
            [
                (now - 10.0, 10.0),
                (now, 0.0),
            ],
            maxlen=100,
        )
        w = est._estimate_wait(0.0)
        assert w is not None
        assert w == 0.0

    # --- zero dt edge case ---

    def test_zero_dt_returns_none(self):
        est = QueueWaitEstimator(strategy="ema")
        now = time.time()
        est._history = deque(
            [(now, 10.0), (now, 10.0)],  # same timestamp → dt=0
            maxlen=100,
        )
        w = est._estimate_wait(10.0)
        assert w is None

    # --- deque overflow (maxlen respected) ---

    def test_window_full_evicts_oldest(self):
        est = QueueWaitEstimator(strategy="ema", fps=10, window_seconds=1)
        # maxlen = 10
        dmap = self._make_dmap(region=(30, 50, 5, 35), density=1.0)
        for _ in range(15):
            est.update(dmap)
        assert len(est._history) == 10  # clamped at maxlen

    # --- reset ---

    def test_reset_clears_history_and_roi(self):
        est = QueueWaitEstimator(strategy="ema")
        dmap = self._make_dmap(region=(30, 50, 5, 35), density=1.0)
        for _ in range(10):
            est.update(dmap)
        assert len(est._history) > 0
        assert est._roi is not None and est._roi.running_mean is not None
        est.reset()
        assert len(est._history) == 0
        assert est._roi.running_mean is None

    # --- all three strategies produce results ---

    @pytest.mark.parametrize("strategy", ["threshold", "morphology", "ema"])
    def test_strategy_produces_wait(self, strategy):
        est = QueueWaitEstimator(strategy=strategy, fps=10, window_seconds=5)
        dmap = self._make_dmap(region=(30, 50, 5, 35), density=1.0)
        # Build history
        w = None
        for _ in range(10):
            w = est.update(dmap)
        assert w is not None
        assert w >= 0

    # --- history property ---

    def test_history_property(self):
        est = QueueWaitEstimator(strategy="ema")
        dmap = self._make_dmap(region=(30, 50, 5, 35), density=1.0)
        est.update(dmap)
        h = est.history
        assert len(h) == 1
        assert isinstance(h[0], tuple)
        assert len(h[0]) == 2  # (timestamp, count)


# ---------------------------------------------------------------------------
# estimate_waiting_time convenience
# ---------------------------------------------------------------------------


class TestEstimateWaitingTime:
    def test_returns_estimator_with_one_frame(self):
        dmap = torch.zeros(64, 64)
        dmap[30:50, 5:35] = 1.0
        est = estimate_waiting_time(dmap, strategy="ema", fps=10)
        assert isinstance(est, QueueWaitEstimator)
        assert len(est.history) == 1

    def test_all_strategies(self):
        dmap = torch.zeros(64, 64)
        dmap[30:50, 5:35] = 1.0
        for s in ["threshold", "morphology", "ema"]:
            est = estimate_waiting_time(dmap, strategy=s)
            assert len(est.history) == 1


# ---------------------------------------------------------------------------
# Integration: QueueWaitEstimator with AdaptiveQueueROI internal state
# ---------------------------------------------------------------------------


class TestROIIntegration:
    def test_ema_strategy_uses_roi(self):
        est = QueueWaitEstimator(strategy="ema")
        assert est._roi is not None
        dmap = torch.ones(32, 32) * 0.5
        # First update → roi learns, queue count estimated
        est.update(dmap)
        # After first frame, roi should have learned the density map
        assert est._roi.running_mean is not None

    def test_non_ema_strategies_skip_roi(self):
        for s in ["threshold", "morphology"]:
            est = QueueWaitEstimator(strategy=s)
            assert est._roi is None
