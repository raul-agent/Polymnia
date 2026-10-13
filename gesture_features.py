#!/usr/bin/env python3
"""Neck-relative BODY_25 wrist features - the pure calculation core.

This module turns dfMaker's person-relative wrist coordinates into bounded,
auditable geometry and timing features. It reads no files and no database: the
public entry point is

    compute_gesture_features(wide_dataframe, config) -> (frame_features, sequence_features)

`gestures.py` (the CLI) is only an adapter that produces the wide frame table
below from the long `multi_data` table and writes the two Parquet files.

Coordinates (see README "Neck-relative gesture features")
---------------------------------------------------------
BODY_25 point 4 is the right wrist and point 7 the left wrist. The values come
from dfMaker's ``nx`` / ``ny`` columns (``fast_scaling = TRUE``, default
``transformation_coords = c(1, 1, 5, 5)``): the origin is the NECK (point 1, not
the nose), X is divided by the neck-to-point-5 X component (BODY_25 point 5 sits
at exactly ``nx = 1.0`` in delivered data; point 5 is ONE shoulder, so this is
not a bilateral shoulder span) and Y is positive UP. So ``ny > 0`` means "above
the neck" and every distance is in neck-space normalized units, not metres.

Input schema (one row per id / people_id / frame)
-------------------------------------------------
``id, people_id, frame, frame_timestamp, nx_4, ny_4, c_4, nx_7, ny_7, c_7``
Extra columns are ignored, and a missing wrist column is filled with NULL (the
other hand still gets its features). ``frame_timestamp`` is elapsed seconds:
known marks must be finite and increase with the frame, while a NULL mark is a
real frame with no clock. It keeps its geometry and its coordinate features and
loses every timed feature; derivatives never cross it, and a clock is only
borrowed across a short internal gap (``timestamp_inferred`` says when).

Pipeline, per (id, people_id) sequence, in this exact order
-----------------------------------------------------------
1. keep a coordinate only when its confidence is a finite probability, > 0 and
   >= threshold; keep a clock only where the source reported a finite one;
2. expand the missing frame indices inside the observed frame span (those rows
   are marked ``frame_inferred``; the span is the *observed* frames, never a
   claim about the whole raw video);
3. inside the grid, linearly interpolate a clock across at most
   ``max_gap_frames`` consecutive untimed internal frames bounded by timed ones
   (marked ``timestamp_inferred``; edges are never extrapolated), and linearly
   interpolate a coordinate across at most ``max_gap_frames`` missing frames
   that have trusted endpoints on both sides and a finite clock span
   (``max_gap_frames`` 0 disables both, long gaps stay NULL);
4. smooth each contiguous run of usable samples with a known clock separately -
   neither an unfilled coordinate gap nor a clock hole is bridged - with an
   adaptive Savitzky-Golay window that fits in the run; sampled uniformly the
   filter runs on the real time step, sampled irregularly the same polynomial
   is fitted against the actual timestamps;
5. velocities and accelerations come from those smoothed coordinates against
   the clock, and nothing timed is invented where the clock is unknown; neck
   distance/angle, above-neck, adjacent-frame step distances and between-hand
   metrics only need trusted coordinates, so they survive an unknown clock.

Missing data is always NULL, never 0 and never inf: geometry needs one valid
sample, velocity two connected samples, acceleration three, a between-hand
metric needs both hands at that frame, and every "time to" / duration statistic
needs a clock on the frame it describes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ----------------------------------------------------------------- vocabulary ----

BODY_25_NECK = 1
BODY_25_RIGHT_WRIST = 4
BODY_25_LEFT_WRIST = 7
HAND_POINTS = {"right": BODY_25_RIGHT_WRIST, "left": BODY_25_LEFT_WRIST}

FEATURE_VERSION = "3"          # bump when the feature meaning changes
BASE_FEATURE_VERSION = "2"     # schema of the default (non-extended) call
PARAM_NAMES = ("min_keypoint_confidence", "max_gap_frames", "window_size", "poly_order")

# The one absolute threshold this module uses. Chosen from the observed speed
# distribution of the delivery corpus (median 0.38-0.39, p90 1.4-1.6 neck-space
# units/s): 0.5 units/s sits at about the 55th percentile, so "moving" means
# faster than the typical frame, not faster than nothing.
MOVEMENT_SPEED = 0.5

# An aggregate needs a denominator. With fewer samples than this the statistic is
# published as NULL - the count columns (n_speed, n_cos_phase) still say what was
# measured. In the 29-clip delivery the phase columns had clips sitting on 1, 5
# and 6 frames, and a "mean" over one frame is noise with three decimals.
MIN_STAT_SAMPLES = 10

REQUIRED_WIDE_COLUMNS = ("id", "people_id", "frame", "frame_timestamp",
                         "nx_4", "ny_4", "c_4", "nx_7", "ny_7", "c_7")

# Per-hand columns, in output order. "(b)" columns are nullable booleans.
HAND_MEASURES = ("confidence", "nx", "ny", "nx_smooth", "ny_smooth",
                 "valid(b)", "observed(b)", "interpolated(b)",
                 "neck_distance", "neck_angle", "above_neck(b)",
                 "velocity_x", "velocity_y", "speed",
                 "accel_x", "accel_y", "accel_magnitude", "step_distance")

KEY_COLUMNS = ("id", "people_id")

FRAME_COLUMNS = (KEY_COLUMNS + ("frame", "frame_timestamp", "source_observed",
                                "frame_inferred", "clock_known", "timestamp_inferred")
                 + tuple(f"{hand}_{m.split('(')[0]}" for hand in ("right", "left") for m in HAND_MEASURES)
                 + ("both_hands_valid", "pair_separation", "pair_angle",
                    "centroid_x", "centroid_y", "mirror_error_x", "mirror_error_y"))

FRAME_BOOL_COLUMNS = tuple(c for c in FRAME_COLUMNS
                           if c not in ("id", "people_id", "frame", "frame_timestamp")
                           and ((c in ("source_observed", "frame_inferred", "clock_known",
                                       "timestamp_inferred", "both_hands_valid"))
                                or c.split("_", 1)[1] in {m.split("(")[0] for m in HAND_MEASURES
                                                          if m.endswith("(b)")}))

SEQUENCE_MEASURES = ("n_observed", "n_interpolated", "n_valid",
                     "frac_observed", "frac_interpolated", "frac_valid",
                     "mean_speed", "max_speed", "mean_accel_magnitude", "max_accel_magnitude",
                     "path_length", "nx_range", "ny_range", "mean_neck_distance",
                     "angle_mean", "angle_std", "time_to_peak_speed",
                     "early_speed_mean", "late_speed_mean")

SEQUENCE_COLUMNS = (KEY_COLUMNS + ("frames_source", "frames_grid", "frames_missing_grid",
                                   "n_timed_frames", "n_timestamp_inferred", "n_missing_timestamps",
                                   "t_start", "t_end", "duration_s")
                    + tuple(f"{hand}_{m}" for hand in ("right", "left") for m in SEQUENCE_MEASURES)
                    + ("n_both_hands", "frac_both_hands", "mean_pair_separation",
                       "max_pair_separation", "mean_mirror_error_x", "mean_mirror_error_y",
                       "time_to_peak_pair_separation"))

# Written next to the outputs so a Parquet file can be read years later.
FEATURE_UNITS = {
    "coordinates (nx, ny, nx_smooth, ny_smooth, centroid_*)": "dfMaker neck-space normalized units, origin BODY_25 neck (point 1), Y positive up",
    "distances (neck_distance, step_distance, pair_separation, path_length, nx_range, ny_range, mirror_error_*)": "normalized neck-space units (NOT metres)",
    "velocity (velocity_x, velocity_y, speed, early/late_speed_mean)": "normalized neck-space units per second",
    "acceleration (accel_x, accel_y, accel_magnitude)": "normalized neck-space units per second squared (NOT joules, no body-mass model)",
    "angles (neck_angle, pair_angle, angle_mean, angle_std)": "radians, atan2(ny, nx); NULL when the vector has zero length",
    "time (t_start, t_end, duration_s, time_to_peak_*, early/late windows)": "elapsed seconds from the source frame_timestamp; NULL where that frame has no clock mark (clock_known = false)",
    "counts and fractions": "float64; fractions divide by frames_grid, every frame of the OBSERVED frame span including frames with no clock mark",
    "NULL": "not measured: absent keypoint, low confidence, or too few connected samples",
}

# ------------------------------------------------- extended block (opt-in) ----
# Added behind a flag so a FEATURE_VERSION 2 read stays byte-identical. Nothing
# here normalizes across clips, nothing here bins space (the observed neck
# distance of a raised hand spans 1.9-2.8 units, so spatial bins saturate), and
# every column is an aggregate of the frame table over finite values only.

EXTENDED_FRAME_MEASURES = ("jerk_x", "jerk_y", "jerk_magnitude")
EXTENDED_FRAME_COLUMNS = tuple(f"{hand}_{m}" for hand in ("right", "left")
                               for m in EXTENDED_FRAME_MEASURES)

_AGG_MEASURES = ("speed", "accel_magnitude", "jerk_magnitude", "neck_distance")
_AGG_STATS = ("min", "p05", "p25", "p50", "p75", "p95", "max", "std")
_SPEED_THRESHOLDS = (0.5, 1.0, 2.0)


def _extended_hand_measures():
    names = [f"{m}_{s}" for m in _AGG_MEASURES for s in _AGG_STATS]
    names += [f"{axis}_{s}" for axis in ("nx", "ny") for s in ("min", "max", "std")]
    names += ["reversals_x", "reversals_y", "reversals_radial", "reversals_per_s"]
    names += ["hull_area", "ellipse_area", "bbox_area"]
    names += ["cv_speed", "mean_over_max", "t_peak_rel", "early_late_speed_ratio",
              "speed_skew", "speed_kurtosis"]
    names += [f"frac_speed_gt_{str(t).replace('.', 'p')}" for t in _SPEED_THRESHOLDS]
    names += ["speed_sq_mean", "speed_sq_sum", "n_speed", "longest_moving_s"]
    return tuple(names)


EXTENDED_BILATERAL_MEASURES = ("cos_phase_mean", "cos_phase_median", "cos_phase_std",
                               "cos_phase_p05", "cos_phase_p95", "frac_cos_gt_0p8",
                               "frac_cos_lt_m0p2", "n_cos_phase",
                               "coactivity_frac_0p5", "coactivity_frac_1p0", "coactivity_bout_n",
                               "coactivity_bout_mean_s", "coactivity_bout_max_s",
                               "mirror_error_x_std", "mirror_error_y_std",
                               "abs_y_pair_sum_mean", "pair_elevation_mean", "bbox_area_wrists")

EXTENDED_HAND_MEASURES = _extended_hand_measures()
EXTENDED_SEQUENCE_MEASURES = (tuple(f"{hand}_{m}" for hand in ("right", "left")
                                    for m in EXTENDED_HAND_MEASURES) + EXTENDED_BILATERAL_MEASURES)

EXTENDED_FEATURE_UNITS = {
    "jerk (jerk_x, jerk_y, jerk_magnitude)": "neck-space units per second cubed; third derivative of the SMOOTHED coordinate on runs of >= 4 timed frames, NULL below that",
    "distribution aggregates (*_min, *_p05, *_p25, *_p50, *_p75, *_p95, *_max, *_std)": "linear-interpolated percentiles and sample std (ddof=1) over finite values of that measure only; *_std is NULL for a single sample",
    "extent (nx_min, nx_max, nx_std, ny_min, ny_max, ny_std)": "smoothed neck-space coordinates; nx_range/ny_range already published are max-min of the same series",
    "reversals (reversals_x, reversals_y, reversals_radial)": ("count of sign changes of THAT component across adjacent timed frames, restricted to samples with |component| > MOVEMENT_SPEED: np.gradient leaves float dust of order 1e-15 on a stationary axis and counting its sign flips would invent a shake. A purely horizontal move therefore reports reversals_x and NULL reversals_y, not 0. Exact-zero samples hold the previous sign. reversals_radial is d(neck_distance)/dt of adjacent timed frames in units/s (never the raw per-step difference, which is a different unit)"),
    "reversals_per_s": "(reversals_x + reversals_y) / duration_s; NULL when either count is NULL or the clock span is NULL or zero",
    "hull_area": "area of the convex hull of the smoothed (nx, ny) points, neck-space units squared; 0 for < 3 points or collinear points",
    "ellipse_area": "area of the 2-standard-deviation ellipse of the smoothed points, 4*pi*sqrt(lambda1*lambda2) of the coordinate covariance; 0 when a principal axis has no spread",
    "bbox_area": "(nx_max - nx_min) * (ny_max - ny_min): area of the axis-aligned bounding box of the trajectory",
    "scale-free intra-clip ratios (cv_speed, mean_over_max, t_peak_rel, early_late_speed_ratio)": ("dimensionless quotients of two statistics of the SAME clip; never a statistic normalized against other clips. t_peak_rel = time_to_peak_speed / duration_s, early_late_speed_ratio = early_speed_mean / late_speed_mean (both v1 columns). NULL when the clip has no sample above MOVEMENT_SPEED: a quotient of float dust is not a shape"),
    "shape of the speed distribution (speed_skew, speed_kurtosis)": "Fisher-Pearson m3/m2^1.5 and excess m4/m2^2 - 3 over finite speed; NULL when fewer than 3 samples, when the variance is 0, or when the clip never exceeds MOVEMENT_SPEED",
    "frac_speed_gt_0p5 / 1p0 / 2p0": "fraction of finite speed samples above ABSOLUTE thresholds of 0.5, 1.0 and 2.0 neck-space units/s (0.5 = MOVEMENT_SPEED, about the 55th percentile of the delivery corpus)",
    "speed_sq_mean, speed_sq_sum": "mean and sum of speed^2, units^2/s^2 (specific kinetic energy is 0.5 * speed_sq_mean, J/kg only if a metre scale and a mass are supplied - they are not)",
    "n_speed": "count of frames with a finite speed, the denominator every speed aggregate above divides by",
    "longest_moving_s": "longest run of ADJACENT timed frames with speed > MOVEMENT_SPEED, measured as last minus first timestamp of the run",
    "cos_phase_*": ("cosine of the angle between the right and left velocity vectors, in [-1, 1]: +1 same direction, -1 opposite. Only frames where BOTH hands exceed MOVEMENT_SPEED enter: two 1e-15 vectors have an undefined angle and would contribute uniform noise. n_cos_phase says how many frames that is - if it is small, the mean is thin"),
    "frac_cos_gt_0p8, frac_cos_lt_m0p2": "fraction of those frames that are in-phase (> 0.8) or genuinely anti-phase (< -0.2)",
    "coactivity_frac_0p5, coactivity_frac_1p0": "fraction of frames with BOTH hands valid and BOTH speeds above the absolute threshold, over frames where both speeds are measurable",
    "coactivity_bout_*": "runs of adjacent timed frames satisfying coactivity; n is always a real count (0.0 = measured and none), mean/max are NULL when there is no bout and when fewer than MIN_STAT_SAMPLES frames had both speeds measurable",
    "mirror_error_x_std, mirror_error_y_std": "sample std of the v1 per-frame mirror errors (|x_R + x_L| about the body midline, |y_R - y_L|)",
    "abs_y_pair_sum_mean": "mean of |y_R + y_L| - the elevation-magnitude quantity, NOT a symmetry measure",
    "pair_elevation_mean": "mean signed (y_R + y_L) / 2: height of the hand pair relative to the neck, positive above",
    "bbox_area_wrists": "|x_R - x_L| * |y_R - y_L|: area of the axis-aligned rectangle having the two wrists as opposite corners; 0 whenever the wrists share an x or a y",
    "NULL in the extended block": ("that measure had no finite sample in this clip, or fewer than "
                                   "MIN_STAT_SAMPLES = 10 samples behind it. A NULL is not a 0, not a "
                                   "small value, and not 'we looked and found nothing': read n_speed "
                                   "and n_cos_phase for the denominator. 10 is a floor, not a claim "
                                   "that a p95 is stable - filter on the published n for that"),
}

EXTENDED_SEMANTICS = {
    "why there are no spatial bins": ("distance to the neck of a raised hand in this corpus spans p10=1.91 to "
                                      "max=2.77 units, so any absolute ring saturates (frac > 3 units is 0 for every "
                                      "clip) and any per-clip quartile ring is uniform by construction. Ring and "
                                      "zone fractions were measured and dropped, not omitted by oversight."),
    "why there is no cross-clip normalization": ("min-max or z-score against the whole clip set makes every feature "
                                                 "dataset-dependent and leaks test clips into training scaling. "
                                                 "Normalization belongs inside a CV pipeline, fit on train only."),
    "why there is no phase-locking value": ("the longest usable contiguous run is about 150 frames (5 s) in 4 clips; "
                                            "a Hilbert phase estimate on that is noise with decimals. Revisit with "
                                            "longer clips and several clips per speaker."),
    "bilateral columns": ("they need both wrists in the same frame. In the 29-clip delivery that is 4 clips; every "
                          "other clip is NULL, and n_both_hands / n_cos_phase say how thin each number is."),
}


def feature_version(extended=False):
    """Schema version of what compute_gesture_features will emit."""
    return FEATURE_VERSION if extended else BASE_FEATURE_VERSION


def sequence_columns(extended=False):
    """Column list compute_gesture_features returns for sequence features."""
    return SEQUENCE_COLUMNS + (EXTENDED_SEQUENCE_MEASURES if extended else ())


SEMANTICS = {
    "reference_origin": ("BODY_25 point 1 (neck) as transformed by dfMaker; never the nose. X is "
                         "normalized by the neck-to-point-5 X component (point 5 is one shoulder, "
                         "not a bilateral shoulder span)"),
    "source_frame_span": ("features cover the observed frame span min(frame)..max(frame) of each "
                          "id/people_id sequence. Frames absent from the source inside that span are "
                          "marked frame_inferred; nothing is claimed about frames outside it, and the "
                          "span is not a claim about the length of the raw video."),
    "clock": ("frame_timestamp NULL means the source gave no clock for that frame: geometry and "
              "coordinate features are kept, timed features are NULL. A missing clock is interpolated "
              "only across at most max_gap_frames consecutive untimed INTERNAL frames bounded by timed "
              "ones (timestamp_inferred); edges are never extrapolated, and known marks must increase "
              "with the frame"),
    "interpolation": ("coordinate gaps of at most max_gap_frames missing frames between valid endpoints "
                      "with a finite clock span are linearly interpolated in elapsed time and flagged "
                      "interpolated; edges are never extrapolated and longer gaps are never bridged"),
    "smoothing": ("adaptive Savitzky-Golay per contiguous run of usable coordinates with a known clock "
                  "(window shrinks to fit the run and its polynomial order with it, window_size=1 "
                  "disables smoothing); unfilled gaps and clock holes are never crossed"),
    "derivatives": "computed from smoothed coordinates against real frame_timestamp seconds, never an assumed frame rate",
    "adjacent_step_distance": ("displacement to the previous frame of a contiguous coordinate run; it "
                               "needs no clock, so path_length stays meaningful where the clock has holes"),
    "angles": ("angle_mean/angle_std are the circular mean and the sqrt(-2 ln R) spread of R = "
               "|mean(exp(i*angle))|; that spread assumes a single concentrated mode (it is a "
               "concentration measure, not a full von Mises fit) and is NULL, together with the mean, "
               "when R is numerically zero and no direction is meaningful"),
    "aggregates": "mean/max over finite values only; NULL when no value exists (a NULL is not a measured 0)",
}


class GestureConfigError(ValueError):
    """The requested features cannot be computed (bad parameter)."""


class GestureDataError(ValueError):
    """The input table cannot be trusted (missing keys, conflicting or impossible rows)."""


# --------------------------------------------------------------------- config ----

_PARAMS = PARAM_NAMES


def _clean_key(name):
    return str(name).strip().replace("-", "_")


def _number(value, name, low, high, integer=False):
    """One finite number inside [low, high] (integral when `integer`)."""
    if isinstance(value, bool) or value is None:
        raise GestureConfigError(f"{name} must be a number, got {value!r}")
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError):
        raise GestureConfigError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(num) or not low <= num <= high:
        raise GestureConfigError(f"{name} must be a finite number in [{low}, {high}], got {value!r}")
    if integer and num != int(num):
        raise GestureConfigError(f"{name} must be a whole number, got {value!r}")
    if integer:
        try:
            return int(num)
        except (TypeError, ValueError, OverflowError):
            raise GestureConfigError(f"{name} is too large to use, got {value!r}") from None
    return num


class GestureConfig:
    """Validated parameters for one feature run."""

    def __init__(self, min_keypoint_confidence=0.2, max_gap_frames=3,
                 window_size=7, poly_order=2):
        self.min_keypoint_confidence = _number(min_keypoint_confidence, "min_keypoint_confidence", 0.0, 1.0)
        self.max_gap_frames = _number(max_gap_frames, "max_gap_frames", 0, 10_000, integer=True)
        window = _number(window_size, "window_size", 1, 10_001, integer=True)
        if window % 2 == 0:
            raise GestureConfigError(f"window_size must be odd (1 disables smoothing), got {window_size!r}")
        self.window_size = window
        self.poly_order = _number(poly_order, "poly_order", 1, 10_000, integer=True)
        if window > 1 and self.poly_order >= window:
            raise GestureConfigError(
                f"poly_order {self.poly_order} must be smaller than window_size {window}; "
                "short runs still shrink the window and its order, and window_size 1 disables smoothing")

    @classmethod
    def from_input(cls, params=None):
        """Build from a mapping, accepting hyphens, rejecting conflicts and typos."""
        if params is None:
            return cls()
        if isinstance(params, GestureConfig):
            return params
        if not hasattr(params, "items"):
            raise GestureConfigError(f"config must be a mapping or GestureConfig, got {params!r}")
        chosen, seen = {}, {}
        for key, value in params.items():
            name = _clean_key(key)
            if name not in _PARAMS:
                raise GestureConfigError(
                    f"unknown gesture parameter {key!r}; supported: {', '.join(_PARAMS)}")
            if name in seen and seen[name] != value:
                raise GestureConfigError(
                    f"conflicting values for {name}: {seen[name]!r} and {value!r}")
            seen[name], chosen[name] = value, value
        return cls(**chosen)

    def as_dict(self):
        return {name: getattr(self, name) for name in _PARAMS}

    def smoothing_params(self, run_length):
        """(window, poly_order) that fit a run; window 1 means "do not smooth".

        ``poly_order`` is already checked against ``window_size``, so an order
        that only has to shrink for a short run can be adapted here.
        """
        window = min(self.window_size, run_length if run_length % 2 else run_length - 1)
        order = min(self.poly_order, window - 1)
        if window < 3 or order < 1:
            return 1, 0
        return window, order


# ------------------------------------------------------------------- plumbing ----

def _runs(mask):
    """Contiguous [start, stop) stretches where `mask` is True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def _is_uniform(times):
    if len(times) < 2:
        return True
    steps = np.diff(times)
    step = float(np.median(steps))
    return step > 0 and float(np.max(np.abs(steps - step))) <= 1e-6 * step


def _local_polyfit(values, times, window, order):
    """Savitzky-Golay for samples that are NOT evenly spaced in time."""
    out = np.empty_like(values)
    half, last = window // 2, len(values) - window
    for i in range(len(values)):
        lo = min(max(i - half, 0), last)
        design = np.vander(times[lo:lo + window] - times[i], order + 1, increasing=True)
        coef = np.linalg.lstsq(design, values[lo:lo + window], rcond=None)[0]
        out[i] = coef[0]
    return out


def _smooth_run(values, times, config):
    """Smooth one contiguous run; never looks outside it."""
    window, order = config.smoothing_params(len(values))
    if window == 1:
        return values.copy()
    if _is_uniform(times):
        return np.asarray(savgol_filter(values, window, order,
                                       delta=float(times[1] - times[0]), mode="interp"), dtype="float64")
    return _local_polyfit(values, times, window, order)


def _derivative(values, times, degree):
    """d/dt (velocity degree 1, acceleration degree 2) or None when too short."""
    if len(values) < degree + 1:
        return None
    out = np.asarray(values, dtype="float64")
    for _ in range(degree):
        out = np.gradient(out, times, edge_order=2 if len(values) >= 3 else 1)
    return out


def _interpolate_gaps(values, times, valid, max_gap):
    """Fill bounded interior gaps in `values` linearly in `times`.

    Returns (filled, interpolated). Gaps touching an edge, longer than `max_gap`,
    or without a finite clock at both endpoints and a positive span stay NULL.
    """
    filled, interpolated = values.copy(), np.zeros(len(values), dtype=bool)
    if max_gap <= 0:
        return filled, interpolated
    known = np.flatnonzero(valid)
    for left, right in zip(known[:-1], known[1:]):
        missing = right - left - 1
        span = times[right] - times[left]
        if not 1 <= missing <= max_gap:
            continue
        if not (np.isfinite(times[left]) and np.isfinite(times[right])) or not span > 0:
            continue                     # no clock, no elapsed time to interpolate on
        between = np.arange(left + 1, right)
        frac = (times[between] - times[left]) / span
        filled[between] = values[left] + frac * (values[right] - values[left])
        interpolated[between] = True
    return filled, interpolated


# ------------------------------------------------------------------- sequences ----

def _validate_input(wide_df):
    """Normalize and trust-check the wide table; returns a sorted copy."""
    if not hasattr(wide_df, "columns"):
        wide_df = pd.DataFrame(wide_df, columns=list(REQUIRED_WIDE_COLUMNS))
    missing_keys = [c for c in ("id", "people_id", "frame") if c not in wide_df.columns]
    if missing_keys:
        raise GestureDataError(f"input is missing required columns: {', '.join(missing_keys)}")
    df = wide_df.copy()
    # A wrist this source never reports is NULL: the other hand keeps its features.
    for column in REQUIRED_WIDE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    df = df.loc[:, list(REQUIRED_WIDE_COLUMNS)]
    if df.empty:
        return df.assign(id=pd.Series(dtype="object"), people_id=pd.Series(dtype="float64"),
                        frame=pd.Series(dtype="int64"), frame_timestamp=pd.Series(dtype="float64"))

    for name in ("id", "people_id", "frame"):
        if df[name].isna().any():
            raise GestureDataError(f"{name} is NULL in {int(df[name].isna().sum())} row(s); "
                                   f"every pose row needs a clip id, person and frame")
    limits = np.iinfo(np.int64)
    # Reject exact Python/NumPy integers before conversion can round them into
    # range (e.g. int64.min - 1 becomes int64.min as a float).
    for value in df["frame"]:
        if isinstance(value, (int, np.integer)) and not limits.min <= int(value) <= limits.max:
            raise GestureDataError("frame must fit int64 frame indices")
    try:
        people = pd.to_numeric(df["people_id"], errors="raise").astype("float64")
        frames = pd.to_numeric(df["frame"], errors="raise")
        stamps = pd.to_numeric(df["frame_timestamp"], errors="raise").astype("float64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise GestureDataError(f"frame / people_id / frame_timestamp must be numeric: {exc}") from None
    if not np.all(np.isfinite(people)):
        raise GestureDataError(f"people_id must be a finite number in every row "
                               f"({int(np.sum(~np.isfinite(people)))} row(s) are not)")
    if not np.all(np.isfinite(frames)):
        raise GestureDataError(f"frame must be a finite integer index "
                               f"({int(np.sum(~np.isfinite(frames)))} row(s) are not)")
    if not np.all(frames == np.floor(frames)):
        raise GestureDataError("frame must hold whole frame indices")
    if pd.api.types.is_integer_dtype(frames.dtype):
        in_range = frames.map(lambda value: limits.min <= int(value) <= limits.max)
    else:
        # float(int64.max) is already 2**63: this upper bound is EXCLUSIVE.
        in_range = (frames >= float(limits.min)) & (frames < float(2**63))
    if not np.all(in_range):
        raise GestureDataError(f"frame must fit int64 frame indices (got values outside "
                               f"[{limits.min}, {limits.max}])")
    # A NULL timestamp is a frame with no clock; a reported one must be a real time.
    reported = ~np.isnan(stamps)
    if not np.all(np.isfinite(stamps[reported])):
        raise GestureDataError(f"frame_timestamp must be finite where reported "
                               f"({int(np.sum(~np.isfinite(stamps[reported])))} row(s) are not)")
    for hand, point in HAND_POINTS.items():
        _check_confidence(df[f"c_{point}"], hand)

    df = df.assign(people_id=people, frame=frames.astype("int64"), frame_timestamp=stamps)
    # A full duplicate row is harmless (re-runs of `merge` append); a duplicate
    # key carrying different values means the frame is ambiguous - fail closed.
    deduped = df.drop_duplicates(keep="first")
    if deduped.duplicated(subset=["id", "people_id", "frame"]).any():
        bad = deduped[deduped.duplicated(subset=["id", "people_id", "frame"], keep=False)]
        keys = ", ".join(f"({r.id}, {r.people_id:g}, {r.frame:d})" for r in bad.head(5).itertuples())
        raise GestureDataError(f"conflicting rows for the same id/people_id/frame: {keys}")
    df = deduped.sort_values(["id", "people_id", "frame"], kind="mergesort").reset_index(drop=True)
    return df


def _check_confidence(values, hand):
    """OpenPose-style confidence: NULL is absent, anything else must be a probability."""
    numbers = values.to_numpy("float64")
    reported = ~np.isnan(numbers)
    if not np.all((numbers[reported] >= 0.0) & (numbers[reported] <= 1.0)):
        bad = numbers[reported][(numbers[reported] < 0.0) | (numbers[reported] > 1.0)][:5]
        raise GestureDataError(f"{hand} wrist confidence must be within [0, 1] where reported "
                               f"(got {', '.join(repr(float(v)) for v in bad)}); "
                               "NULL is treated as absent, an impossible value is not")


def _sequence_grid(grp, label, config):
    """Frame grid, clock, clock-known / inferred flags and source rows over the span.

    A clock is only borrowed across at most `max_gap_frames` consecutive untimed
    INTERNAL frames bounded by timed ones; edges are never extrapolated.
    """
    frames = grp["frame"].to_numpy("int64")
    marks = grp["frame_timestamp"].to_numpy("float64")
    if np.any(np.diff(frames) <= 0):
        raise GestureDataError(f"{label}: repeated frame index inside the sequence")
    reported = np.isfinite(marks)
    if reported.any() and np.any(np.diff(marks[reported]) <= 0):
        raise GestureDataError(f"{label}: frame_timestamp must increase with the frame wherever "
                               "a timestamp is reported, so elapsed time is well defined")

    grid = np.arange(int(frames[0]), int(frames[-1]) + 1, dtype="int64")
    times = np.full(len(grid), np.nan, dtype="float64")
    times[frames - frames[0]] = marks
    inferred = np.zeros(len(grid), dtype=bool)
    timed = np.flatnonzero(np.isfinite(times))
    if config.max_gap_frames > 0:
        for left, right in zip(timed[:-1], timed[1:]):
            if not 1 <= right - left - 1 <= config.max_gap_frames:
                continue
            between = np.arange(left + 1, right)
            frac = (grid[between] - grid[left]) / (grid[right] - grid[left])
            times[between] = times[left] + frac * (times[right] - times[left])
            inferred[between] = True

    observed = np.zeros(len(grid), dtype=bool)
    observed[frames - frames[0]] = True
    return grid, times, np.isfinite(times), inferred, observed


def _hand_arrays(grp, hand, size, first_frame):
    """Raw nx / ny / confidence placed on the frame grid (NaN where no row)."""
    point = HAND_POINTS[hand]
    out = []
    for field in ("nx", "ny", "c"):
        arr = np.full(size, np.nan, dtype="float64")
        arr[grp["frame"].to_numpy("int64") - first_frame] = grp[f"{field}_{point}"].to_numpy("float64")
        out.append(arr)
    return out


def _hand_rows(nx_raw, ny_raw, conf, times, clock, observed, config):
    """All per-hand frame features for one sequence, as a dict of grids."""
    usable = (np.isfinite(nx_raw) & np.isfinite(ny_raw) & np.isfinite(conf)
              & (conf > 0.0) & (conf >= config.min_keypoint_confidence))
    nx_fill, interp_x = _interpolate_gaps(nx_raw, times, usable, config.max_gap_frames)
    ny_fill, _ = _interpolate_gaps(ny_raw, times, usable, config.max_gap_frames)
    valid = usable | interp_x
    # A coordinate that arrived with low confidence is still not usable: report
    # NULL, never the value nobody trusted.
    nx_usable = np.where(valid, nx_fill, np.nan)
    ny_usable = np.where(valid, ny_fill, np.nan)

    out = {"confidence": conf, "valid": valid, "observed": usable & observed,
           "interpolated": interp_x, "nx": nx_usable, "ny": ny_usable}
    # Geometry follows trusted coordinates; smoothing and derivatives only run
    # where that coordinate also has a clock, so a clock hole stops the physics
    # without losing the position.
    sx, sy = nx_usable.copy(), ny_usable.copy()
    velocity = {k: np.full(len(times), np.nan) for k in ("velocity_x", "velocity_y", "speed")}
    accel = {k: np.full(len(times), np.nan) for k in ("accel_x", "accel_y", "accel_magnitude")}
    jerk = {k: np.full(len(times), np.nan) for k in EXTENDED_FRAME_MEASURES}
    step = np.full(len(times), np.nan)

    for start, stop in _runs(valid & clock):
        run_t, run_x, run_y = times[start:stop], sx[start:stop], sy[start:stop]
        smooth_x, smooth_y = _smooth_run(run_x, run_t, config), _smooth_run(run_y, run_t, config)
        sx[start:stop], sy[start:stop] = smooth_x, smooth_y
        velocity_x, velocity_y = _derivative(smooth_x, run_t, 1), _derivative(smooth_y, run_t, 1)
        if velocity_x is not None and velocity_y is not None:
            velocity["velocity_x"][start:stop] = velocity_x
            velocity["velocity_y"][start:stop] = velocity_y
            velocity["speed"][start:stop] = np.hypot(velocity_x, velocity_y)
        ax, ay = _derivative(smooth_x, run_t, 2), _derivative(smooth_y, run_t, 2)
        if ax is not None and ay is not None:
            accel["accel_x"][start:stop], accel["accel_y"][start:stop] = ax, ay
            accel["accel_magnitude"][start:stop] = np.hypot(ax, ay)
        jx, jy = _derivative(smooth_x, run_t, 3), _derivative(smooth_y, run_t, 3)
        if jx is not None and jy is not None:
            jerk["jerk_x"][start:stop], jerk["jerk_y"][start:stop] = jx, jy
            jerk["jerk_magnitude"][start:stop] = np.hypot(jx, jy)

    # Adjacent-frame displacement needs two neighbouring trusted coordinates, not
    # a clock, so path_length stays meaningful across a hole in the timestamps.
    for start, stop in _runs(valid):
        if stop - start >= 2:
            step[start + 1:stop] = np.hypot(np.diff(sx[start:stop]), np.diff(sy[start:stop]))

    distance = np.hypot(sx, sy)
    angle = np.where(distance > 0, np.arctan2(sy, sx), np.nan)
    above_neck = np.full(len(times), None, dtype=object)
    known = np.isfinite(sy)
    above_neck[known] = sy[known] > 0
    out.update({"nx_smooth": sx, "ny_smooth": sy, "neck_distance": distance,
                "neck_angle": angle, "above_neck": above_neck,
                "velocity_x": velocity["velocity_x"], "velocity_y": velocity["velocity_y"],
                "speed": velocity["speed"], "accel_x": accel["accel_x"], "accel_y": accel["accel_y"],
                "accel_magnitude": accel["accel_magnitude"], "step_distance": step})
    out.update(jerk)
    return out


def _sequence_frame_rows(clip_id, people, grp, config, extended=False):
    """Every grid row of one (id, people_id) sequence as a list of dicts."""
    grid, times, clock, clock_inferred, observed = _sequence_grid(
        grp, f"sequence {clip_id} / person {people:g}", config)
    measures = HAND_MEASURES + (EXTENDED_FRAME_MEASURES if extended else ())
    hands = {hand: _hand_rows(*_hand_arrays(grp, hand, len(grid), grid[0]),
                              times, clock, observed, config)
             for hand in ("right", "left")}

    both = hands["right"]["valid"] & hands["left"]["valid"]
    rx, ry = hands["right"]["nx_smooth"], hands["right"]["ny_smooth"]
    lx, ly = hands["left"]["nx_smooth"], hands["left"]["ny_smooth"]
    separation = np.where(both, np.hypot(rx - lx, ry - ly), np.nan)
    pair = {
        "both_hands_valid": both,
        "pair_separation": separation,
        "pair_angle": np.where(both & (separation > 0), np.arctan2(ry - ly, rx - lx), np.nan),
        "centroid_x": np.where(both, (rx + lx) / 2.0, np.nan),
        "centroid_y": np.where(both, (ry + ly) / 2.0, np.nan),
        "mirror_error_x": np.where(both, np.abs(rx + lx), np.nan),
        "mirror_error_y": np.where(both, np.abs(ry - ly), np.nan),
    }

    rows = []
    for i, frame in enumerate(grid):
        row = {"id": clip_id, "people_id": people, "frame": int(frame),
               "frame_timestamp": times[i], "source_observed": bool(observed[i]),
               "frame_inferred": not bool(observed[i]), "clock_known": bool(clock[i]),
               "timestamp_inferred": bool(clock_inferred[i])}
        for hand in ("right", "left"):
            for measure in measures:
                name = measure.split("(")[0]
                row[f"{hand}_{name}"] = hands[hand][name][i]
        row.update({name: values[i] for name, values in pair.items()})
        rows.append(row)
    return rows


# ------------------------------------------------------------------ aggregates ----

def _finite(values):
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


def _stat(values, reducer, default=np.nan):
    values = _finite(values)
    return float(reducer(values)) if values.size else default


def _clock_span(times, clock):
    """(first, last) known clock of a sequence, or (NULL, NULL) when there is none."""
    known = np.asarray(times, "float64")[np.asarray(clock, dtype=bool)]
    if not known.size:
        return np.nan, np.nan
    return float(known.min()), float(known.max())


def _time_to_peak(values, times, clock, t_start):
    """Elapsed time from the start of the known clock to the highest value.

    When the winning frame has no clock the answer is NULL: silently reporting
    the second-highest timed value would describe a different peak.
    """
    values, times = np.asarray(values, "float64"), np.asarray(times, "float64")
    trusted = np.asarray(clock, dtype=bool) & np.isfinite(values) & np.isfinite(times)
    peak = np.flatnonzero(np.isfinite(values))
    if not peak.size:
        return np.nan
    winner = peak[np.argmax(values[peak])]
    if not trusted[winner] or not np.isfinite(t_start):
        return np.nan
    return float(times[winner] - t_start)


def _edge_window_mean(values, times, clock, t_start, t_end, first):
    """Mean of the finite values inside the first/last 10% of the known clock span."""
    values, times = np.asarray(values, "float64"), np.asarray(times, "float64")
    trusted = np.asarray(clock, dtype=bool) & np.isfinite(times)
    if not np.isfinite(t_start) or not np.isfinite(t_end):
        return np.nan
    edge = 0.1 * (t_end - t_start)
    inside = trusted & ((times <= t_start + edge) if first else (times >= t_end - edge))
    return _stat(values[inside], np.mean)


def _circular_stats(angles, zero_resultant=1e-12):
    """Circular mean and the sqrt(-2 ln R) spread of the resultant R.

    That spread is a concentration measure for a single mode (not a full von
    Mises fit). When R is numerically zero no direction is meaningful, so both
    the mean and the spread are NULL rather than an arbitrary angle.
    """
    angles = _finite(angles)
    if not angles.size:
        return np.nan, np.nan
    mean_vector = complex(float(np.cos(angles).mean()), float(np.sin(angles).mean()))
    resultant = min(abs(mean_vector), 1.0)
    if resultant <= zero_resultant:
        return np.nan, np.nan
    mean = math.atan2(mean_vector.imag, mean_vector.real)
    return mean, math.sqrt(max(-2.0 * math.log(resultant), 0.0))


def _std(values, ddof=1):
    """Sample std, NULL for fewer than two samples (a single sample has no spread)."""
    values = _finite(values)
    return float(np.std(values, ddof=ddof)) if values.size >= 2 else np.nan


def _percentile(values, q):
    values = _finite(values)
    return float(np.percentile(values, q)) if values.size else np.nan


def _distribution_block(prefix, values):
    """min / percentiles / max / std of one measure, over its finite samples only."""
    values = _finite(values)
    if values.size < MIN_STAT_SAMPLES:
        return {f"{prefix}_{suffix}": np.nan
                for suffix in ("min", "p05", "p25", "p50", "p75", "p95", "max", "std")}
    block = {f"{prefix}_min": float(values.min()), f"{prefix}_max": float(values.max()),
             f"{prefix}_std": _std(values)}
    for q in (5, 25, 50, 75, 95):
        block[f"{prefix}_p{q:02d}"] = float(np.percentile(values, q))
    return block


def _sign_changes(values):
    """Count of sign changes across the given samples, or NULL for too few of them.

    Exact zeros hold the previous sign instead of inventing a reversal, so a
    hand that parks at x=0 does not look like it is shaking.
    """
    signs = np.sign(_finite(values))
    signs = signs[signs != 0]
    if signs.size < MIN_STAT_SAMPLES:
        return np.nan
    return float(np.count_nonzero(np.diff(signs)))


def _convex_hull_area(x, y):
    """Area of the convex hull (Andrew's monotone chain).

    The sample floor counts FRAMES, not distinct positions: a hand parked in one
    spot for 61 frames has a swept area of 0, which is a measurement, while 3
    frames have not measured an area at all. Duplicates are removed with exact
    equality, never rounding: rounding would turn a straight trajectory into a
    thin fake polygon with a nonzero area.
    """
    x, y = _finite(x), _finite(y)
    if x.size < MIN_STAT_SAMPLES:
        return np.nan
    points = np.unique(np.column_stack([x, y]), axis=0)
    if len(points) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    chains = []
    for order in (points, points[::-1]):
        chain = []
        for point in order:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        chains.append(chain)
    hull = np.array(chains[0][:-1] + chains[1][:-1])
    if len(hull) < 3:
        return 0.0
    x_y = np.column_stack([hull[:, 0], hull[:, 1]])
    return float(0.5 * abs(np.dot(x_y[:, 0], np.roll(x_y[:, 1], -1))
                           - np.dot(x_y[:, 1], np.roll(x_y[:, 0], -1))))


def _ellipse_area(x, y):
    """Area of the 2-standard-deviation ellipse, 4*pi*sqrt(lambda1*lambda2).

    rowvar=False because the columns are the variables: the rows are frames. The
    floor counts frames: a position held for 61 frames encloses 0 area, which is
    a measurement.
    """
    x, y = _finite(x), _finite(y)
    if x.size < MIN_STAT_SAMPLES:
        return np.nan
    if x.size < 2:
        return 0.0
    covariance = np.cov(np.column_stack([x, y]), rowvar=False, ddof=1)
    eigenvalues = np.clip(np.linalg.eigvalsh(np.atleast_2d(covariance)), 0.0, None)
    return float(4.0 * math.pi * math.sqrt(float(eigenvalues.prod())))


def _run_durations(mask, times, clock, frame_numbers):
    """(n, mean_s, max_s) of runs of ADJACENT timed frames satisfying mask.

    The count is always a real count: 0.0 means "measured, and there was no such
    run", which is a measurement. mean/max are NULL when there is nothing to
    average - never 0.0 seconds.
    """
    adjacent = np.concatenate(([True], np.diff(frame_numbers) == 1)) & np.concatenate(
        ([True], clock[1:] & clock[:-1]))
    durations = []
    for start, stop in _runs(np.asarray(mask, dtype=bool) & np.asarray(clock, dtype=bool) & adjacent):
        run = np.asarray(times, "float64")[start:stop]
        run = run[np.isfinite(run)]
        if run.size:
            durations.append(float(run[-1] - run[0]))
    if not durations:
        return 0.0, np.nan, np.nan
    return float(len(durations)), float(np.mean(durations)), float(np.max(durations))


def _shape_moments(values):
    """Fisher-Pearson skew and excess kurtosis; NULL when the shape is undefined."""
    values = _finite(values)
    if values.size < 3:
        return np.nan, np.nan
    centered = values - values.mean()
    second = float((centered ** 2).mean())
    if second <= 0.0:
        return np.nan, np.nan
    return (float((centered ** 3).mean() / second ** 1.5),
            float((centered ** 4).mean() / second ** 2 - 3.0))


def _hand_extended_summary(frames, hand, base):
    """Per-hand aggregates that only look at this clip."""
    out = {}

    def column(name):
        return frames[f"{hand}_{name}"].to_numpy("float64")

    times = frames["frame_timestamp"].to_numpy("float64")
    clock = frames["clock_known"].to_numpy("bool")
    frame_numbers = frames["frame"].to_numpy("int64")
    speed = _finite(column("speed"))
    speed_arr = column("speed")
    has_movement = bool(np.any(speed_arr > MOVEMENT_SPEED))
    enough_speed = speed.size >= MIN_STAT_SAMPLES

    for measure in _AGG_MEASURES:
        out.update(_distribution_block(f"{hand}_{measure}", column(measure)))
    for axis in ("nx", "ny"):
        values = _finite(column(f"{axis}_smooth"))
        enough = values.size >= MIN_STAT_SAMPLES
        out[f"{hand}_{axis}_min"] = float(values.min()) if enough else np.nan
        out[f"{hand}_{axis}_max"] = float(values.max()) if enough else np.nan
        out[f"{hand}_{axis}_std"] = _std(values) if enough else np.nan

    timed = clock & np.isfinite(times)
    # A stationary hand still gets a derivative of order 1e-15 from the
    # second-order edge formula, and that dust flips sign. A direction only
    # counts where THAT component is measurable above MOVEMENT_SPEED, so a purely
    # horizontal move reports reversals in x and NULL in y, never 0 fake flips.
    def axis_reversals(component):
        values = column(component)
        return _sign_changes(values[timed & (np.abs(values) > MOVEMENT_SPEED)])

    out[f"{hand}_reversals_x"] = axis_reversals("velocity_x")
    out[f"{hand}_reversals_y"] = axis_reversals("velocity_y")
    distance, frames_ahead = column("neck_distance"), np.diff(frame_numbers) == 1
    pair_timed = frames_ahead & clock[1:] & clock[:-1]
    dt = np.diff(times)
    with np.errstate(divide="ignore", invalid="ignore"):
        # A difference per STEP is not a rate: divide by the real elapsed time so
        # this compares against a per-second threshold in the same units as speed.
        radial = np.diff(distance) / dt
    radial = np.where(pair_timed & np.isfinite(dt) & (dt > 0), radial, np.nan)
    out[f"{hand}_reversals_radial"] = _sign_changes(radial[np.abs(radial) > MOVEMENT_SPEED])
    duration = base.get("duration_s", np.nan)
    reversals = [out[f"{hand}_reversals_x"], out[f"{hand}_reversals_y"]]
    out[f"{hand}_reversals_per_s"] = (float(np.nansum(reversals)) / float(duration)
                                      if np.isfinite(duration) and duration > 0
                                      and np.any(np.isfinite(reversals)) else np.nan)

    smooth_x, smooth_y = column("nx_smooth"), column("ny_smooth")
    both_axes = np.isfinite(smooth_x) & np.isfinite(smooth_y)
    if int(both_axes.sum()) >= MIN_STAT_SAMPLES:
        out[f"{hand}_hull_area"] = _convex_hull_area(smooth_x[both_axes], smooth_y[both_axes])
        out[f"{hand}_ellipse_area"] = _ellipse_area(smooth_x[both_axes], smooth_y[both_axes])
        out[f"{hand}_bbox_area"] = (float(smooth_x[both_axes].max() - smooth_x[both_axes].min())
                                    * float(smooth_y[both_axes].max() - smooth_y[both_axes].min()))
    else:
        out[f"{hand}_hull_area"] = out[f"{hand}_ellipse_area"] = out[f"{hand}_bbox_area"] = np.nan

    mean_speed = float(speed.mean()) if speed.size else np.nan
    max_speed = base.get(f"{hand}_max_speed", np.nan)
    out[f"{hand}_cv_speed"] = (float(_std(speed) / mean_speed) if enough_speed and has_movement
                               and np.isfinite(mean_speed) and mean_speed > 0 else np.nan)
    out[f"{hand}_mean_over_max"] = (float(mean_speed / max_speed) if enough_speed
                                    and np.isfinite(mean_speed) and np.isfinite(max_speed)
                                    and max_speed > 0 else np.nan)
    peak, duration = base.get(f"{hand}_time_to_peak_speed", np.nan), base.get("duration_s", np.nan)
    out[f"{hand}_t_peak_rel"] = (float(peak / duration) if enough_speed and np.isfinite(peak)
                                 and np.isfinite(duration) and duration > 0 else np.nan)
    early, late = base.get(f"{hand}_early_speed_mean", np.nan), base.get(f"{hand}_late_speed_mean", np.nan)
    out[f"{hand}_early_late_speed_ratio"] = (float(early / late) if enough_speed and has_movement
                                             and np.isfinite(early) and np.isfinite(late)
                                             and late > 0 else np.nan)
    skew, kurtosis = _shape_moments(speed)
    out[f"{hand}_speed_skew"] = skew if has_movement and enough_speed else np.nan
    out[f"{hand}_speed_kurtosis"] = kurtosis if has_movement and enough_speed else np.nan

    for threshold in _SPEED_THRESHOLDS:
        name = f"{hand}_frac_speed_gt_{str(threshold).replace('.', 'p')}"
        out[name] = float((speed > threshold).mean()) if enough_speed else np.nan
    out[f"{hand}_speed_sq_mean"] = float((speed ** 2).mean()) if enough_speed else np.nan
    out[f"{hand}_speed_sq_sum"] = float((speed ** 2).sum()) if enough_speed else np.nan
    out[f"{hand}_n_speed"] = float(speed.size)
    out[f"{hand}_longest_moving_s"] = (_run_durations(column("speed") > MOVEMENT_SPEED,
                                                      times, clock, frame_numbers)[2]
                                       if enough_speed else np.nan)
    return out


def _bilateral_extended_summary(frames):
    """Two-hand measures. NULL in every clip where the two wrists are not both tracked."""
    out = {}
    times = frames["frame_timestamp"].to_numpy("float64")
    clock = frames["clock_known"].to_numpy("bool")
    frame_numbers = frames["frame"].to_numpy("int64")
    vectors = {hand: (frames[f"{hand}_velocity_x"].to_numpy("float64"),
                      frames[f"{hand}_velocity_y"].to_numpy("float64")) for hand in ("right", "left")}
    speeds = {hand: frames[f"{hand}_speed"].to_numpy("float64") for hand in ("right", "left")}

    rx, ry = vectors["right"]
    lx, ly = vectors["left"]
    moving = (np.isfinite(speeds["right"]) & (speeds["right"] > MOVEMENT_SPEED)
              & np.isfinite(speeds["left"]) & (speeds["left"] > MOVEMENT_SPEED))
    norm = np.hypot(rx, ry) * np.hypot(lx, ly)
    usable = moving & np.isfinite(norm) & (norm > 0.0)
    cos_phase = np.divide(rx * lx + ry * ly, norm, out=np.full(len(norm), np.nan), where=usable)
    finite_cos = _finite(cos_phase)
    out["n_cos_phase"] = float(finite_cos.size)
    thin_cos = ["cos_phase_mean", "cos_phase_median", "cos_phase_std", "cos_phase_p05",
                "cos_phase_p95", "frac_cos_gt_0p8", "frac_cos_lt_m0p2"]
    if finite_cos.size >= MIN_STAT_SAMPLES:
        out["cos_phase_mean"] = float(finite_cos.mean())
        out["cos_phase_median"] = float(np.percentile(finite_cos, 50))
        out["cos_phase_std"] = _std(finite_cos)
        out["cos_phase_p05"] = float(np.percentile(finite_cos, 5))
        out["cos_phase_p95"] = float(np.percentile(finite_cos, 95))
        out["frac_cos_gt_0p8"] = float((finite_cos > 0.8).mean())
        out["frac_cos_lt_m0p2"] = float((finite_cos < -0.2).mean())
    else:
        for name in thin_cos:
            out[name] = np.nan

    measurable = np.isfinite(speeds["right"]) & np.isfinite(speeds["left"])
    enough_both = int(measurable.sum()) >= MIN_STAT_SAMPLES
    active = {"0p5": measurable & (speeds["right"] > MOVEMENT_SPEED)
                            & (speeds["left"] > MOVEMENT_SPEED),
              "1p0": measurable & (speeds["right"] > 1.0) & (speeds["left"] > 1.0)}
    for suffix, mask in active.items():
        out[f"coactivity_frac_{suffix}"] = (float(mask[measurable].mean())
                                            if enough_both else np.nan)
    count, mean_s, max_s = _run_durations(active["0p5"], times, clock, frame_numbers)
    out["coactivity_bout_n"] = count
    out["coactivity_bout_mean_s"] = mean_s if enough_both else np.nan
    out["coactivity_bout_max_s"] = max_s if enough_both else np.nan

    for name, source in (("mirror_error_x_std", "mirror_error_x"),
                         ("mirror_error_y_std", "mirror_error_y")):
        values = _finite(frames[source].to_numpy("float64"))
        out[name] = _std(values) if values.size >= MIN_STAT_SAMPLES else np.nan

    both = frames["both_hands_valid"].to_numpy("bool")
    sx = {hand: frames[f"{hand}_nx_smooth"].to_numpy("float64") for hand in ("right", "left")}
    sy = {hand: frames[f"{hand}_ny_smooth"].to_numpy("float64") for hand in ("right", "left")}
    paired = (both & np.isfinite(sx["right"]) & np.isfinite(sx["left"])
              & np.isfinite(sy["right"]) & np.isfinite(sy["left"]))
    if int(paired.sum()) >= MIN_STAT_SAMPLES:
        y_sum = sy["right"][paired] + sy["left"][paired]
        out["abs_y_pair_sum_mean"] = float(np.abs(y_sum).mean())
        out["pair_elevation_mean"] = float((y_sum / 2.0).mean())
        out["bbox_area_wrists"] = float((np.abs(sx["right"][paired] - sx["left"][paired])
                                         * np.abs(sy["right"][paired] - sy["left"][paired])).mean())
    else:
        out["abs_y_pair_sum_mean"] = out["pair_elevation_mean"] = out["bbox_area_wrists"] = np.nan
    return out


def _hand_summary(frames, hand, frames_grid):
    def column(name):
        return frames[f"{hand}_{name}"].to_numpy("float64")

    def count(name):
        return float(frames[f"{hand}_{name}"].to_numpy("bool").sum())

    n_observed, n_interp, n_valid = count("observed"), count("interpolated"), count("valid")
    angle_mean, angle_std = _circular_stats(column("neck_angle"))
    times = frames["frame_timestamp"].to_numpy("float64")
    clock = frames["clock_known"].to_numpy("bool")
    t_start, t_end = _clock_span(times, clock)
    speed = column("speed")
    return {
        f"{hand}_n_observed": n_observed, f"{hand}_n_interpolated": n_interp,
        f"{hand}_n_valid": n_valid,
        f"{hand}_frac_observed": n_observed / frames_grid,
        f"{hand}_frac_interpolated": n_interp / frames_grid,
        f"{hand}_frac_valid": n_valid / frames_grid,
        f"{hand}_mean_speed": _stat(speed, np.mean),
        f"{hand}_max_speed": _stat(speed, np.max),
        f"{hand}_mean_accel_magnitude": _stat(column("accel_magnitude"), np.mean),
        f"{hand}_max_accel_magnitude": _stat(column("accel_magnitude"), np.max),
        f"{hand}_path_length": _stat(column("step_distance"), np.sum),
        f"{hand}_nx_range": _stat(column("nx_smooth"), lambda a: a.max() - a.min()),
        f"{hand}_ny_range": _stat(column("ny_smooth"), lambda a: a.max() - a.min()),
        f"{hand}_mean_neck_distance": _stat(column("neck_distance"), np.mean),
        f"{hand}_angle_mean": angle_mean, f"{hand}_angle_std": angle_std,
        f"{hand}_time_to_peak_speed": _time_to_peak(speed, times, clock, t_start),
        f"{hand}_early_speed_mean": _edge_window_mean(speed, times, clock, t_start, t_end, True),
        f"{hand}_late_speed_mean": _edge_window_mean(speed, times, clock, t_start, t_end, False),
    }


def _sequence_summary(clip_id, people, frames, extended=False):
    times = frames["frame_timestamp"].to_numpy("float64")
    clock = frames["clock_known"].to_numpy("bool")
    both = frames["both_hands_valid"].to_numpy("bool")
    separation = frames["pair_separation"].to_numpy("float64")
    n_both = float(both.sum())
    frames_grid = float(len(frames))
    n_timed = float(clock.sum())
    t_start, t_end = _clock_span(times, clock)
    summary = {
        "id": clip_id, "people_id": people,
        "frames_source": float(frames["source_observed"].to_numpy("bool").sum()),
        "frames_grid": frames_grid,
        "frames_missing_grid": float(frames["frame_inferred"].to_numpy("bool").sum()),
        "n_timed_frames": n_timed,
        "n_timestamp_inferred": float(frames["timestamp_inferred"].to_numpy("bool").sum()),
        "n_missing_timestamps": frames_grid - n_timed,
        "t_start": t_start, "t_end": t_end,
        "duration_s": t_end - t_start if np.isfinite(t_start) and np.isfinite(t_end) else np.nan,
        "n_both_hands": n_both, "frac_both_hands": n_both / frames_grid,
        "mean_pair_separation": _stat(separation, np.mean),
        "max_pair_separation": _stat(separation, np.max),
        "mean_mirror_error_x": _stat(frames["mirror_error_x"].to_numpy("float64"), np.mean),
        "mean_mirror_error_y": _stat(frames["mirror_error_y"].to_numpy("float64"), np.mean),
        "time_to_peak_pair_separation": _time_to_peak(separation, times, clock, t_start),
    }
    for hand in ("right", "left"):
        summary.update(_hand_summary(frames, hand, frames_grid))
    if extended:
        for hand in ("right", "left"):
            summary.update(_hand_extended_summary(frames, hand, summary))
        summary.update(_bilateral_extended_summary(frames))
    return summary


# ----------------------------------------------------------------- public API ----

def compute_gesture_features(wide_df, config=None, extended=False):
    """Neck-relative wrist features for a wide per-frame table.

    Args:
        wide_df: rows of REQUIRED_WIDE_COLUMNS (never modified).
        config: GestureConfig, a mapping of GestureConfig parameters, or None.
        extended: add the FEATURE_VERSION 3 block - jerk in the frame table and
            per-clip distribution, shape, direction and two-hand aggregates in
            the sequence table. Nothing in it looks at other clips.

    Returns:
        (frame_features, sequence_features) DataFrames with FRAME_COLUMNS
        (+EXTENDED_FRAME_COLUMNS) and SEQUENCE_COLUMNS (+EXTENDED_SEQUENCE_MEASURES)
        - the same schema even when nothing was measurable.
    """
    cfg = config if isinstance(config, GestureConfig) else GestureConfig.from_input(config)
    df = _validate_input(wide_df)
    frame_columns = FRAME_COLUMNS + (EXTENDED_FRAME_COLUMNS if extended else ())
    seq_columns = sequence_columns(extended)

    records = []
    for (clip_id, people), grp in df.groupby(["id", "people_id"], sort=True):
        records.extend(_sequence_frame_rows(clip_id, people, grp, cfg, extended=extended))

    frames = _typed_frame(pd.DataFrame(records, columns=list(frame_columns)),
                          extended=extended)
    if frames.empty:
        return frames, _typed_frame(pd.DataFrame(columns=list(seq_columns)),
                                    sequence=True, extended=extended)

    summaries = [_sequence_summary(clip_id, people, grp, extended=extended)
                 for (clip_id, people), grp in frames.groupby(["id", "people_id"], sort=True)]
    return frames, _typed_frame(pd.DataFrame(summaries, columns=list(seq_columns)),
                                sequence=True, extended=extended)


def _typed_frame(df, sequence=False, extended=False):
    """Fixed dtypes: float64 features (NULL = NaN), nullable bool, int64 frame."""
    if sequence:
        columns = list(sequence_columns(extended))
    else:
        columns = list(FRAME_COLUMNS + (EXTENDED_FRAME_COLUMNS if extended else ()))
    out = pd.DataFrame(index=range(len(df)), columns=columns)
    for name in out.columns:
        values = df[name].to_numpy(dtype=object) if len(df) else np.empty(0, dtype=object)
        if name == "id":
            out[name] = pd.Series(values, dtype="object")
        elif name == "frame":
            out[name] = pd.Series(pd.array(values, dtype="Int64"), dtype="Int64")
        elif name in FRAME_BOOL_COLUMNS and not sequence:
            out[name] = pd.Series(pd.array([None if v is None or (isinstance(v, float) and math.isnan(v))
                                            else bool(v) for v in values], dtype="boolean"),
                                  dtype="boolean")
        else:
            numbers = np.full(len(values), np.nan, dtype="float64")
            for i, value in enumerate(values):
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    numbers[i] = float(value)
            out[name] = np.where(np.isfinite(numbers), numbers, np.nan) if len(values) else numbers
    return out
