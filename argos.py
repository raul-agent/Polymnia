#!/usr/bin/env python3

import argparse
import math
import os
import subprocess
import sys

import body_visibility as bv

# The gate counts BODY_25 point indices (4 = right wrist, 7 = left wrist).
# BODY_25B has the same array length but maps the indices differently, so the
# gated path pins the model explicitly instead of trusting OpenPose's default.
BODY_25_MODEL = "BODY_25"
R_SCRIPT = "max_people_classification.R"


# openpose_process runs openpose on a video and saves the results to the output_folder/dataset/OpenPose/video_name
# it also runs max_people_classification.R on the json_folder and saves the results to the output_folder/dataset/video_name 
# folder structure:
# output_folder/
#   dataset/
#     OpenPose/
#       video_name/
#         JSON_FILES/
#         skeletons/
#         previous_json/          # only with --filter-body-visibility
#     1_persons/
#       video_name/
#         parquet_files/
#     2_persons/
#       video_name/
#         parquet_files/
#     ...
#     body_visibility/            # only with --filter-body-visibility
#       video_name.json
#       rejected/<reason>/parquet_files/
def openpose_argv(video, json_folder, skeletons_folder, face_hands, skeletons, model_pose=None):
    """The openpose.bin command line for one video (same flags as before)."""
    argv = ["build/examples/openpose/openpose.bin", "--video", video,
            "--write_json", json_folder]
    if skeletons_folder:
        argv += ["--write_video", f"{skeletons_folder}/skeleton.avi"]
    argv += ["--display", "0", "--render_pose", "1" if skeletons_folder else "0"]
    if model_pose:
        argv += ["--model_pose", model_pose]
    if face_hands:
        argv += ["--face", "--hand"]
    return argv


def run_openpose(video, output_folder, openpose_path, face_hands, skeletons,
                 model_pose=None, call=None):
    """Run OpenPose on one video and write its JSON under the dataset folder."""
    video_name = clip_name(video)
    json_folder = f"{output_folder}/dataset/OpenPose/{video_name}/JSON_FILES"
    os.makedirs(json_folder, exist_ok=True)

    skeletons_folder = None
    if skeletons:
        skeletons_folder = f"{output_folder}/dataset/OpenPose/{video_name}/skeletons"
        os.makedirs(skeletons_folder, exist_ok=True)

    argv = openpose_argv(video, json_folder, skeletons_folder, face_hands, skeletons, model_pose)
    # OpenPose resolves its model files relative to its own folder, which is what
    # the previous `cd openpose_path && openpose.bin ...` did.
    return call(argv, cwd=openpose_path)


def run_df_maker(json_folder, dataset_dir, call=None):
    """max_people_classification.R: dfMaker + classification into {n}_persons/."""
    print(f"Running {R_SCRIPT} on {json_folder}")
    # the R script reads config_dfMaker.json from the working directory, so this
    # stays a relative call, exactly as before
    return call(["Rscript", R_SCRIPT, json_folder, dataset_dir], quiet=True)


def clip_name(video):
    return os.path.basename(video)[:-len(".mp4")] if video.endswith(".mp4") else os.path.basename(video)


def video_frame_count(video):
    """Frame count from the container metadata, or 0 when it cannot be read."""
    import cv2  # local import: the gate is optional, the legacy path needs no cv2
    capture = cv2.VideoCapture(video)
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return count if count > 0 else 0


def _frame_count(video):
    """(expected_frames, error) from the container metadata.

    The gate needs the frame count before it can measure anything, and the
    reader can fail in more ways than "returns 0": a decoder that throws, a NaN
    or inf count, a count too large to convert. All of those are unusable data
    for *this* clip and are reported like any other data error, so the run keeps
    going with the later clips instead of dying on a traceback.
    """
    try:
        count = video_frame_count(video)
    except Exception as exc:  # noqa: BLE001 - video reader failures, not a bug in the gate
        return 0, f"cannot read the frame count of {video}: {type(exc).__name__}: {exc}"
    try:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"not an integer frame count: {count!r}")
        if count <= 0 or not math.isfinite(float(count)):
            raise ValueError(f"no usable frame count: {count!r}")
    except (OverflowError, TypeError, ValueError) as exc:
        return 0, f"unusable frame count for {video}: {exc}"
    return count, None


def _default_call():
    """The production seam: run argv, with the same cwd/quiet semantics as before."""
    def call(argv, cwd=None, quiet=False):
        kwargs = {"cwd": cwd} if cwd else {}
        if quiet:  # the R step is chatty; it always ran with its output dropped
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL
        return subprocess.run(argv, **kwargs).returncode
    return call


def _wrap_call(runner):
    """Adapt a test runner `runner(argv, cwd=None, quiet=False)` to the seam."""
    def call(argv, cwd=None, quiet=False):
        return runner(argv, cwd=cwd, quiet=quiet)
    return call


def _gate_config(args):
    if not getattr(args, "filter_body_visibility", False):
        return None
    return bv.VisibilityConfig.from_args(
        getattr(args, "required_body_points", bv.DEFAULT_REQUIRED_BODY_POINTS),
        getattr(args, "min_visible_percent", bv.DEFAULT_MIN_VISIBLE_PERCENT),
        getattr(args, "min_keypoint_confidence", bv.DEFAULT_MIN_KEYPOINT_CONFIDENCE),
    )


def _report_paths(output_folder, video_name):
    base = f"{output_folder}/dataset/body_visibility"
    return os.path.join(base, f"{video_name}.json")


def process_video(video, output_folder, args, config, call):
    """One clip: OpenPose, the optional BODY_25 gate, then dfMaker. 0 = keep going."""
    dataset_dir = f"{output_folder}/dataset"
    if config is None:
        run_openpose(video, output_folder, args.openpose_path, args.face_hands, args.skeletons,
                     call=call)
        run_df_maker(f"{dataset_dir}/OpenPose/{clip_name(video)}/JSON_FILES", dataset_dir, call=call)
        return 0

    video_name = clip_name(video)
    json_folder = f"{dataset_dir}/OpenPose/{video_name}/JSON_FILES"
    clip_dir = os.path.dirname(json_folder)
    report = _report_paths(output_folder, video_name)

    def fail(stage, message, archive_json=False):
        """Report an unusable measurement, keep the clip, quarantine its Parquet.

        The report and the quarantine are the whole point of this function, so a
        failure of the optional JSON archiving is folded into the reported
        message instead of skipping them: a stale Parquet must never survive an
        error just because moving JSON went wrong.
        """
        if archive_json:
            try:
                bv.archive_json_dir(json_folder, os.path.join(clip_dir, "previous_json"))
            except OSError as exc:
                message += (f"; previous OpenPose JSON could not be archived "
                            f"({type(exc).__name__}: {exc})")
        print(f"❌ {video_name}: {message}", file=sys.stderr)
        bv.atomic_write_json(report, {
            "clip": video_name, "decision": bv.REASON_DATA_ERROR, "accepted": False,
            "error": message, "stage": stage, "expected_frames": expected,
            "config": config.as_dict(),
        })
        bv.quarantine_stale_parquet(dataset_dir, video_name, "error")
        return 1

    expected, metadata_error = _frame_count(video)
    if metadata_error:
        return fail("video_metadata", metadata_error)

    # Whatever a previous run left here would be indistinguishable from the new
    # frames, so it is preserved elsewhere first: the gate only ever measures
    # OpenPose output produced by this invocation.
    try:
        archived = bv.archive_json_dir(json_folder, os.path.join(clip_dir, "previous_json"))
    except OSError as exc:
        # archiving must not short-circuit the report and the quarantine
        return fail("previous_json", f"cannot archive previous OpenPose JSON of {video_name}: "
                                     f"{type(exc).__name__}: {exc}")
    if archived:
        print(f"   archived {len(archived)} JSON files from a previous run of {video_name}")

    try:
        rc = run_openpose(video, output_folder, args.openpose_path, args.face_hands,
                          args.skeletons, model_pose=BODY_25_MODEL, call=call)
    except OSError as exc:
        # no returncode at all: the binary or its working folder is unusable
        return fail("openpose", f"OpenPose could not be started ({type(exc).__name__}: {exc}); "
                                f"no JSON output was produced", archive_json=True)
    if rc != 0:
        return fail("openpose", f"OpenPose failed (exit {rc}); its JSON output is incomplete",
                    archive_json=True)

    decision = bv.evaluate_body_visibility(json_folder, expected, config, clip=video_name)
    bv.atomic_write_json(report, decision.as_dict())
    if decision.error:
        bv.quarantine_stale_parquet(dataset_dir, video_name, "error")
        try:
            bv.archive_json_dir(json_folder, os.path.join(clip_dir, "previous_json"))
        except OSError as exc:
            # already reported and quarantined; keep the later clips of this run
            print(f"⚠️  {video_name}: could not archive its OpenPose JSON "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)
        print(f"❌ {video_name}: untrustworthy OpenPose output: {decision.error}", file=sys.stderr)
        return 1
    if not decision.accepted:
        moved = bv.quarantine_stale_parquet(dataset_dir, video_name, decision.reason)
        print(f"🚫 {video_name}: rejected ({decision.reason}) — {_coverage_text(decision)}; "
              f"video and JSON kept, {len(moved)} stale parquet moved aside")
        return 0

    try:
        rc = run_df_maker(json_folder, dataset_dir, call=call)
    except OSError as exc:
        # the gate passed but the dataframe step never started: an accepted
        # report must not stand on top of a missing dataframe
        return fail("df_maker", f"{R_SCRIPT} could not be started ({type(exc).__name__}: {exc}) "
                                f"after a passed visibility gate")
    if rc != 0:
        # the clip passed the gate but the dataframe is missing: do not leave the
        # accepted verdict (or a stale Parquet) as if the stage had succeeded
        return fail("df_maker", f"{R_SCRIPT} failed (exit {rc}) after a passed visibility gate")
    print(f"✅ {video_name}: gate passed — {_coverage_text(decision)}")
    return 0


def _coverage_text(decision):
    return ", ".join(f"BODY_25[{p['point']}] {p['visible_frames']}/{p['denominator_frames']} "
                     f"= {p['coverage_percent']:.1f}%" for p in decision.points)


def build_parser():
    parser = argparse.ArgumentParser(description="Run OpenPose and dfMaker on a folder of videos. Video names must be in the format: 2017-07-21_0000_US_KABC_Eyewitness_News_356.505_361.531_before.mp4")
    parser.add_argument('--videos_folder', help='folder with videos to run OpenPose and dfMaker on', required=True)
    parser.add_argument('--output_folder', help='output folder to save the dataframes', required=True)
    parser.add_argument('--openpose_path', help='path to OpenPose', default='/opt/openpose')
    parser.add_argument('--face_hands', help='run OpenPose with face and hands', default=True)
    parser.add_argument('--skeletons', help='run OpenPose with skeletons', default=True)
    parser.add_argument('--filter-body-visibility', dest='filter_body_visibility',
                        action='store_true',
                        help='drop clips where the requested BODY_25 joints are not visible '
                             'long enough (disabled by default; needs OpenPose --model_pose BODY_25)')
    parser.add_argument('--required-body-points', dest='required_body_points', default='4,7',
                        help='comma-separated BODY_25 indices that must be visible, 0-24 '
                             '(default: 4,7 = both wrists)')
    parser.add_argument('--min-visible-percent', dest='min_visible_percent', type=float, default=40.0,
                        help='minimum percentage of ALL video frames each point must be visible in, '
                             'independently per point (default: 40)')
    parser.add_argument('--min-keypoint-confidence', dest='min_keypoint_confidence', type=float,
                        default=0.2,
                        help='minimum OpenPose score for a point to count as visible, 0-1 (default: 0.2)')
    return parser


def run(args, runner=None):
    """Do the work; returns the process exit code (0 = nothing failed)."""
    try:
        config = _gate_config(args)
    except bv.VisibilityConfigError as exc:
        print(f"body visibility: {exc}", file=sys.stderr)
        return 2

    # if the videos_folder doesn't exist create it
    if not os.path.exists(args.videos_folder):
        print('Folder with videos doesn\'t exist.')
        return 0

    # if the output_folder doesn't exist create it
    os.makedirs(args.output_folder, exist_ok=True)

    call = _default_call() if runner is None else _wrap_call(runner)
    status = 0
    # loop through all the videos in the videos_folder
    for video in sorted(os.listdir(args.videos_folder)):
        # check if the video is an mp4
        if video.endswith('.mp4'):
            # get the absolute path of the video
            full_video_path = os.path.abspath(f'{args.videos_folder}/{video}')
            full_output_path = os.path.abspath(args.output_folder)
            print(f'Processing {full_video_path}')
            status = max(status, process_video(full_video_path, full_output_path, args,
                                               config, call))
    return status


def main():
    args = build_parser().parse_args()
    sys.exit(run(args))


if __name__ == '__main__':
    main()
