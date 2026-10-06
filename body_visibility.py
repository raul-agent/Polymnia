#!/usr/bin/env python3
"""Optional BODY_25 visibility gate for POLYMNIA clips.

The gate answers one question per clip: are the requested BODY_25 points
reliably visible for enough of the clip to be worth keeping? It reads the
OpenPose `JSON_FILES` output of a clip and the frame count of the *source*
video, and counts, independently for every requested point, in how many of
ALL the video frames that point is visible with confidence >= threshold. The
windows where each point is visible do not have to coincide: a clip where the
right wrist is visible in the first half and the left wrist in the second
half passes at 40%, even though they are never visible together.

Decisions come in three flavours:

* accepted            - every point clears `min_visible_percent`
* low_coverage        - measurable, but some point does not
* multi_person        - some frame reports more than one person (OpenPose
                        output is ambiguous for a "single speaker" dataset)

Anything that makes the measurement untrustworthy is a `data_error` (fail
closed): unparsable JSON, non-finite coordinates, a skeleton that is not
BODY_25, duplicated or out-of-range frame indices. A data error is not a
judgement about the clip, so it is reported separately and must abort the run
instead of quietly dropping the clip.

No GPU, no OpenPose, no R: the evaluator only reads JSON files, so it is fully
testable with synthetic frames (see tools/test_body_visibility.py).

OpenPose JSON layout (BODY_25, `--model_pose BODY_25`): one file per frame,
`{clip}_{frame_index:012d}_keypoints.json`, 0-based frame index, with
`people[i]["pose_keypoints_2d"]` as 75 floats (25 points x x, y, score). A
score of 0 means "not detected" and the x/y around it are placeholders.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile

BODY_25_POINTS = 25
BODY_25_VALUES = BODY_25_POINTS * 3  # x, y, score per point

DEFAULT_REQUIRED_BODY_POINTS = (4, 7)  # BODY_25 right wrist, left wrist
DEFAULT_MIN_VISIBLE_PERCENT = 40.0
DEFAULT_MIN_KEYPOINT_CONFIDENCE = 0.2

REASON_ACCEPTED = "accepted"
REASON_LOW_COVERAGE = "low_coverage"
REASON_MULTI_PERSON = "multi_person"
REASON_DATA_ERROR = "data_error"

JSON_SUFFIX = "_keypoints.json"
FRAME_NAME = re.compile(r"^(?P<clip>.+)_(?P<index>\d{12})" + re.escape(JSON_SUFFIX) + "$")


class VisibilityConfigError(ValueError):
    """The gate was asked for something it cannot measure."""


class OpenPoseDataError(ValueError):
    """The OpenPose output is not trustworthy enough to measure anything.

    The error carries what is still known about the clip. `stats` holds the
    frame inventory (`with_json`, the valid JSON files found) and how far
    counting got (`processed`, `no_person`, `multi_person`); `counts` the
    per-point tallies. `stats["inventory_known"]` says whether the file names
    themselves could be trusted, which decides whether presence is a number or
    simply unknown.
    """

    stats = {}
    counts = {}


class VisibilityConfig:
    """Validated gate parameters (BODY_25 indices, percentages, confidence)."""

    def __init__(self, required_points, min_visible_percent, min_keypoint_confidence):
        self.required_points = tuple(required_points)
        self.min_visible_percent = float(min_visible_percent)
        self.min_keypoint_confidence = float(min_keypoint_confidence)

    @classmethod
    def from_args(cls, required_points, min_visible_percent, min_keypoint_confidence):
        """Build from CLI-ish values, raising VisibilityConfigError if invalid."""
        return cls(
            _parse_points(required_points),
            _parse_float(min_visible_percent, "min_visible_percent", 0.0, 100.0),
            _parse_float(min_keypoint_confidence, "min_keypoint_confidence", 0.0, 1.0),
        )

    def as_dict(self):
        return {
            "required_points": list(self.required_points),
            "min_visible_percent": self.min_visible_percent,
            "min_keypoint_confidence": self.min_keypoint_confidence,
        }


def _parse_float(value, name, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise VisibilityConfigError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(number) or not low <= number <= high:
        raise VisibilityConfigError(f"{name} must be within [{low}, {high}], got {value!r}")
    return number


def _parse_points(value):
    """BODY_25 indices from a comma-separated list or an iterable of ints."""
    items = str(value).split(",") if isinstance(value, str) else list(value)
    points = []
    for item in items:
        try:
            point = int(str(item).strip())
        except (TypeError, ValueError):
            raise VisibilityConfigError(f"body point indices must be integers, got {item!r}")
        if not 0 <= point < BODY_25_POINTS:
            raise VisibilityConfigError(f"body point index {point} is outside BODY_25 0..{BODY_25_POINTS - 1}")
        if point in points:
            raise VisibilityConfigError(f"duplicated body point index {point}")
        points.append(point)
    if not points:
        raise VisibilityConfigError("at least one required body point is needed")
    return tuple(points)


class VisibilityDecision:
    """The outcome of one clip evaluation, ready to be serialised as a report."""

    def __init__(self, clip, expected_frames, config):
        self.clip = clip
        self.expected_frames = expected_frames
        self.config = config
        self.accepted = False
        self.reason = REASON_DATA_ERROR
        self.error = None
        self.frames_with_json = 0
        self.frames_missing_json = expected_frames
        self.frames_no_person = 0
        self.frames_multi_person = 0
        self.frames_processed = 0
        self.statistics_complete = True
        self.points = []

    def as_dict(self):
        """The report. Presence/absence are `None` when the inventory failed.

        A stopped run says so with `statistics_complete: false` plus
        `frames_processed`; a completed one keeps the keys it always had.
        """
        report = {
            "clip": self.clip,
            "decision": self.reason,
            "accepted": self.accepted,
            "error": self.error,
            "expected_frames": self.expected_frames,
            "frames_with_json": self.frames_with_json,
            "frames_missing_json": self.frames_missing_json,
            "frames_no_person": self.frames_no_person,
            "frames_multi_person": self.frames_multi_person,
            "statistics_complete": self.statistics_complete,
            "points": self.points,
            "config": self.config.as_dict(),
        }
        if not self.statistics_complete:
            report["frames_processed"] = self.frames_processed
        return report


def count_keypoint_frames(json_dir, expected_frames, required_points, min_keypoint_confidence):
    """Per-point visible-frame counts over `expected_frames` video frames.

    Returns (counts, stats). Frames without a JSON file, frames with no person
    and frames where the point has score < threshold all count as absent.
    Raises OpenPoseDataError when the output cannot be trusted.
    """
    counts = {point: 0 for point in required_points}
    # `with_json` is the inventory of valid frame files, taken before any of them
    # is parsed: counting can stop halfway through a folder whose files all
    # exist, and none of them may then be reported as "missing JSON".
    stats = {"with_json": 0, "no_person": 0, "multi_person": 0, "processed": 0,
             "inventory_known": False}
    try:
        frames = _frame_files(json_dir, expected_frames)
        stats["inventory_known"] = True
        stats["with_json"] = len(frames)
        _count_into(counts, stats, frames, required_points, min_keypoint_confidence)
    except OpenPoseDataError as exc:
        exc.stats, exc.counts = dict(stats), dict(counts)
        raise
    return counts, stats


def _count_into(counts, stats, frames, required_points, min_keypoint_confidence):
    for _index, path in frames:
        _count_frame(counts, stats, path, required_points, min_keypoint_confidence)
        stats["processed"] += 1


def _count_frame(counts, stats, path, required_points, min_keypoint_confidence):
    """Add one frame to the tallies, raising OpenPoseDataError if it is garbage."""
    name = os.path.basename(path)
    payload = _load_json(path)
    people = _people(payload, name)
    if len(people) > 1:
        # an ambiguous frame is a frame with JSON: it belongs under multi_person,
        # never folded into the missing files of the report
        stats["multi_person"] += 1
        return
    if not people:
        stats["no_person"] += 1
        return
    keypoints = _body_keypoints(people[0], name)
    for point in required_points:
        score = keypoints[3 * point + 2]  # already checked finite and in range
        if score <= 0.0:
            continue  # OpenPose writes 0 for "not detected", even at threshold 0
        if score >= min_keypoint_confidence:
            counts[point] += 1


def _frame_files(json_dir, expected_frames):
    """Sorted (frame_index, path) pairs, validated against the video.

    The inventory is complete or it is nothing: an unreadable folder, a
    duplicated index, an index beyond the video or an unexpected name all raise,
    so the caller reports presence as unknown instead of guessing a count.
    """
    if not os.path.isdir(json_dir):
        raise OpenPoseDataError(f"OpenPose JSON folder not found: {json_dir}")
    try:
        names = sorted(os.listdir(json_dir))
    except OSError as exc:
        raise OpenPoseDataError(f"cannot list the OpenPose JSON folder {json_dir}: {exc}")
    found = {}
    for name in names:
        if not name.endswith(JSON_SUFFIX):
            continue
        match = FRAME_NAME.match(name)
        if not match:
            raise OpenPoseDataError(f"unexpected OpenPose JSON name: {name}")
        index = int(match.group("index"))
        if index in found:
            raise OpenPoseDataError(f"duplicated frame index {index} in OpenPose JSON output")
        if index >= expected_frames:
            raise OpenPoseDataError(
                f"OpenPose JSON frame index {index} is beyond the video frame count {expected_frames}"
            )
        found[index] = os.path.join(json_dir, name)
    return sorted(found.items())


def _load_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise OpenPoseDataError(f"invalid JSON in {os.path.basename(path)}: {exc}")
    except UnicodeDecodeError as exc:
        # a truncated or binary file is unparsable data, not a Python crash
        raise OpenPoseDataError(
            f"{os.path.basename(path)} is not UTF-8 text: {exc}")
    except ValueError as exc:
        # `json` raises a *plain* ValueError (not JSONDecodeError) for a number
        # with more digits than CPython's int <-> str limit: the text is valid
        # JSON the parser refuses to turn into data. The global digit limit is a
        # memory guard and stays as it is; here it is simply untrustworthy input.
        raise OpenPoseDataError(
            f"unparsable number in {os.path.basename(path)}: {exc}")
    except OSError as exc:
        raise OpenPoseDataError(f"cannot read {path}: {exc}")


def _people(payload, name):
    if not isinstance(payload, dict) or not isinstance(payload.get("people"), list):
        raise OpenPoseDataError(f"missing 'people' list in {name}")
    return payload["people"]


def _body_keypoints(person, name):
    """The 75 floats of one BODY_25 person, validated as measurement data.

    `float(x)` alone is not enough: it happily accepts `true`, `"0.9"` and an
    out-of-range score as confidence, and raises OverflowError on an integer too
    large to convert. A score is only a measurement when it is a JSON number
    between 0 and 1, so anything else fails closed.
    """
    keypoints = person.get("pose_keypoints_2d") if isinstance(person, dict) else None
    if not isinstance(keypoints, list) or len(keypoints) != BODY_25_VALUES:
        raise OpenPoseDataError(
            f"{name}: pose_keypoints_2d is not a BODY_25 skeleton "
            f"(expected {BODY_25_VALUES} floats, got "
            f"{len(keypoints) if isinstance(keypoints, list) else type(keypoints).__name__})"
        )
    values = []
    for position, raw in enumerate(keypoints):
        # bool is an int subclass but never a measurement; strings are not numbers
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise OpenPoseDataError(
                f"{name}: pose_keypoints_2d[{position}] is not a JSON number "
                f"({type(raw).__name__})"
            )
        try:
            value = float(raw)
        except (OverflowError, ValueError):
            raise OpenPoseDataError(
                f"{name}: pose_keypoints_2d[{position}] cannot be represented as a number"
            )
        if not math.isfinite(value):
            raise OpenPoseDataError(f"{name}: non-finite value in pose_keypoints_2d")
        values.append(value)
    for point in range(BODY_25_POINTS):
        score = values[3 * point + 2]
        if not 0.0 <= score <= 1.0:
            raise OpenPoseDataError(
                f"{name}: BODY_25[{point}] confidence {score} is outside [0, 1]"
            )
    return values


def evaluate_body_visibility(json_dir, expected_frames, config, clip=None):
    """Measure one clip. Never raises for clip data problems: see `decision.error`."""
    if isinstance(expected_frames, bool) or not isinstance(expected_frames, int) or expected_frames <= 0:
        raise VisibilityConfigError(f"expected_frames must be a positive int, got {expected_frames!r}")
    if not isinstance(config, VisibilityConfig):
        raise VisibilityConfigError("config must be a VisibilityConfig")
    clip = clip or os.path.basename(os.path.normpath(os.path.dirname(json_dir))) or "clip"
    decision = VisibilityDecision(clip, expected_frames, config)
    try:
        counts, stats = count_keypoint_frames(
            json_dir, expected_frames, config.required_points, config.min_keypoint_confidence
        )
    except OpenPoseDataError as exc:
        decision.error = str(exc)
        decision.reason = REASON_DATA_ERROR
        decision.statistics_complete = False
        decision.frames_processed = int((getattr(exc, "stats", None) or {}).get("processed", 0))
        _apply_partial_inventory(decision, getattr(exc, "stats", None) or {})
        return decision

    decision.frames_with_json = stats["with_json"]
    decision.frames_no_person = stats["no_person"]
    decision.frames_multi_person = stats["multi_person"]
    decision.frames_missing_json = expected_frames - stats["with_json"]
    decision.points = [
        {
            "point": point,
            "visible_frames": counts[point],
            "denominator_frames": expected_frames,
            "coverage_percent": counts[point] * 100.0 / expected_frames,
        }
        for point in config.required_points
    ]

    if decision.frames_multi_person:
        decision.reason = REASON_MULTI_PERSON
        return decision

    # Both sides are scaled by the denominator, so the exact-threshold boundary
    # stays inclusive without comparing two independently rounded percentages.
    low = [
        p["point"] for p in decision.points
        if p["visible_frames"] * 100.0 < config.min_visible_percent * expected_frames
    ]
    decision.accepted = not low
    decision.reason = REASON_ACCEPTED if decision.accepted else REASON_LOW_COVERAGE
    return decision


def _apply_partial_inventory(decision, seen):
    """Fill the frame statistics of a run that stopped at a data error.

    What was counted before the bad frame is still a fact, but only when the
    file inventory itself finished: if the names could not be trusted (missing
    folder, duplicated or out-of-range index, unreadable directory), stating how
    many files are missing would be a guess, so both counts stay unknown.
    """
    decision.frames_no_person = int(seen.get("no_person", 0))
    decision.frames_multi_person = int(seen.get("multi_person", 0))
    if not seen.get("inventory_known"):
        decision.frames_with_json = None
        decision.frames_missing_json = None
        return
    decision.frames_with_json = int(seen.get("with_json", 0))
    decision.frames_missing_json = decision.expected_frames - decision.frames_with_json


def atomic_write_json(path, payload):
    """Write a report through a temp file + rename, so no half-written report."""
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def _unique_path(path):
    """path, path.1, path.2 ... first name that does not exist yet."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}.{n}{ext}"):
        n += 1
    return f"{stem}.{n}{ext}"


def quarantine_stale_parquet(dataset_dir, clip, category):
    """Move `dataset/*_persons/parquet_files/{clip}.parquet` out of the way.

    A rejected clip must not leave a Parquet behind that a later `voice`/`merge`
    picks up. Only files of this clip move; they land under
    `dataset/body_visibility/rejected/<category>/parquet_files/`, which is not a
    `*_persons` folder, so nothing downstream looks there. Names are made unique
    instead of overwritten. Returns the moved destinations.
    """
    files = os.path.join(dataset_dir, "body_visibility", "rejected", category, "parquet_files")
    moved = []
    if not os.path.isdir(dataset_dir):
        return moved
    for persons in sorted(os.listdir(dataset_dir)):
        if not persons.endswith("_persons"):
            continue
        source = os.path.join(dataset_dir, persons, "parquet_files", f"{clip}.parquet")
        if not os.path.isfile(source):
            continue
        target = _unique_path(os.path.join(files, f"{clip}__{persons}.parquet"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(source, target)
        moved.append(target)
    return moved


def archive_json_dir(json_dir, dest_dir):
    """Move existing OpenPose JSON of a clip out of `json_dir`, keeping it.

    An enabled rerun must not blend a previous (possibly truncated) OpenPose run
    into the new one: the gate needs the JSON of *this* run. The old files are
    preserved under `dest_dir` (never deleted), so `json_dir` is empty and the
    next OpenPose call is the only source of frames. Returns moved file names.
    """
    if not os.path.isdir(json_dir):
        return []
    stale = [n for n in sorted(os.listdir(json_dir)) if n.endswith(JSON_SUFFIX)]
    if not stale:
        return []
    os.makedirs(dest_dir, exist_ok=True)
    moved = []
    for name in stale:
        target = _unique_path(os.path.join(dest_dir, name))
        os.replace(os.path.join(json_dir, name), target)
        moved.append(os.path.basename(target))
    for name in ("num_persons.txt",):
        source = os.path.join(json_dir, name)
        if os.path.isfile(source):
            os.replace(source, _unique_path(os.path.join(dest_dir, name)))
    return moved
