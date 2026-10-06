"""Tests for the optional BODY_25 gate inside `argos` (argos.py).

These are integration tests at the CLI boundary: the OpenPose binary and the R
dfMaker step are replaced by a fake command runner, so nothing needs a GPU,
OpenPose or R. Videos are tiny real mp4 files written with OpenCV, so the frame
count that feeds the gate comes from genuine video metadata.

What is pinned here:
  * disabled (default) behaviour is the legacy one: OpenPose then R, no gate
    artefacts, previous JSON untouched;
  * enabled + accepted: OpenPose runs with an explicit BODY_25 model, R runs,
    the Parquet stays, a report is written;
  * enabled + rejected: R never runs, the report is written, media and JSON are
    kept, and a stale Parquet from an earlier run is moved out of `*_persons`;
  * enabled + external failure: reported as an error, the run exits non-zero and
    no stale JSON/R output is used;
  * enabled reruns archive the previous JSON so a truncated run cannot be
    measured on frames from the previous one.
"""
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argos  # noqa: E402
import body_visibility as bv  # noqa: E402

FAIL = []
CLIP = "2017_01_01_US_TEST_Clips_1.0_2.0_before"


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------- fixtures ----

def make_video(path, frames):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def write_openpose_json(json_dir, clip, frames, windows={4: (0, 100), 7: (0, 100)}, conf=0.9):
    os.makedirs(json_dir, exist_ok=True)
    for idx in range(frames):
        body = [0.0] * 75
        for point, (start, stop) in windows.items():
            if start <= idx < stop:
                body[3 * point:3 * point + 3] = [float(idx), 1.0, conf]
        payload = {"version": "1.5.1", "people": [{"pose_keypoints_2d": body}]}
        with open(os.path.join(json_dir, f"{clip}_{idx:012d}_keypoints.json"), "w") as fh:
            json.dump(payload, fh)


def write_parquet(dataset_dir, persons, clip, tag="stale"):
    folder = os.path.join(dataset_dir, f"{persons}_persons", "parquet_files")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{clip}.parquet")
    with open(path, "w") as fh:
        fh.write(tag)
    return path


class LaunchFailTools:
    """Fake tools whose external process cannot even be started (OSError).

    FileNotFoundError is what `subprocess.run` raises when openpose.bin (or its
    cwd) is missing; PermissionError is what it raises for a non-executable
    Rscript. Both used to escape the gate and abort the whole run.
    """

    def __init__(self, frames, fail_on, error, fail_once=False, r_rc=0):
        self.frames = frames
        self.fail_on = fail_on  # "openpose" or "Rscript"
        self.error = error
        # with fail_once only the first clip fails, so a later clip can show that
        # the run went on instead of aborting
        self.fail_once = fail_once
        self.r_rc = r_rc
        self.argvs = []
        self.r_runs = 0
        self.seen = []

    def __call__(self, argv, cwd=None, quiet=False):
        self.argvs.append(list(argv))
        who = "openpose" if argv[0].endswith("openpose.bin") else "Rscript"
        self.seen.append(who)
        already_failed = self.seen.count(who) > 1
        if who == self.fail_on and not (self.fail_once and already_failed):
            raise self.error(f"{argv[0]}: {self.error.__name__}")
        if who == "openpose":
            write_openpose_json(argv[argv.index("--write_json") + 1], CLIP, self.frames,
                                {4: (0, 100), 7: (0, 100)})
            return 0
        self.r_runs += 1
        write_parquet(argv[3], 1, os.path.basename(os.path.dirname(argv[2])), tag="fresh")
        return self.r_rc


class FakeTools:
    """Stand-in for openpose.bin and Rscript, driven through the runner hook."""

    def __init__(self, frames, windows=None, conf=0.9, openpose_rc=0, r_rc=0, multi_person=False):
        self.frames = frames
        self.windows = windows if windows is not None else {4: (0, 100), 7: (0, 100)}
        self.conf = conf
        self.openpose_rc = openpose_rc
        self.r_rc = r_rc
        self.multi_person = multi_person
        self.argvs = []
        self.cwd = None
        self.model_pose = None
        self.r_runs = 0

    def __call__(self, argv, cwd=None, quiet=False):
        self.argvs.append(list(argv))
        self.cwd = cwd
        if argv[0].endswith("openpose.bin"):
            json_dir = argv[argv.index("--write_json") + 1]
            write_openpose_json(json_dir, CLIP, self.frames, self.windows, self.conf)
            if self.multi_person:
                payload = {"version": "1.5.1", "people": [
                    {"pose_keypoints_2d": [0.9] * 75}, {"pose_keypoints_2d": [0.9] * 75}]}
                with open(os.path.join(json_dir, f"{CLIP}_{0:012d}_keypoints.json"), "w") as fh:
                    json.dump(payload, fh)
            return self.openpose_rc
        if argv[0] == "Rscript":  # max_people_classification.R -> parquet
            self.r_runs += 1
            script, json_folder, dataset_dir = argv[1], argv[2], argv[3]
            assert script.endswith("max_people_classification.R"), script
            clip = os.path.basename(os.path.dirname(json_folder))  # .../OpenPose/<clip>/JSON_FILES
            write_parquet(dataset_dir, 1, clip, tag="fresh")
            with open(os.path.join(json_folder, "num_persons.txt"), "w") as fh:
                fh.write("1\n")
            return self.r_rc
        raise AssertionError(f"unexpected command: {argv[:3]}")


def make_layout(frames=10):
    root = tempfile.mkdtemp(prefix="argos-bv-")
    videos, output = os.path.join(root, "videos"), os.path.join(root, "out")
    os.makedirs(videos)
    actual = make_video(os.path.join(videos, f"{CLIP}.mp4"), frames)
    os.makedirs(output)
    return root, videos, output, actual


def gate_args(videos, output, **over):
    """Parse argos CLI flags the way the wrapper forwards them."""
    argv = ["--videos_folder", videos, "--output_folder", output, "--filter-body-visibility"]
    for key, value in over.items():
        argv += [f"--{key}", str(value)]
    return argos.build_parser().parse_args(argv)


def run_gate(videos, output, tools=None, **over):
    args = gate_args(videos, output, **over)
    tools = tools or FakeTools(10)
    rc = argos.run(args, runner=tools)
    return rc, tools


def report(output):
    path = os.path.join(output, "dataset", "body_visibility", f"{CLIP}.json")
    return json.load(open(path)) if os.path.isfile(path) else None


def json_dir(output, clip=CLIP):
    return os.path.join(output, "dataset", "OpenPose", clip, "JSON_FILES")


def report_for(output, clip):
    path = os.path.join(output, "dataset", "body_visibility", f"{clip}.json")
    return json.load(open(path)) if os.path.isfile(path) else None


# ------------------------------------------------------------ disabled ----

def test_disabled_matches_legacy():
    root, videos, output, frames = make_layout(10)
    tools = FakeTools(frames)
    write_parquet(os.path.join(output, "dataset"), 1, CLIP, tag="old")
    rc = argos.run(argos.build_parser().parse_args(
        ["--videos_folder", videos, "--output_folder", output]), runner=tools)
    check("disabled: OpenPose + R, exito", rc == 0 and tools.r_runs == 1)
    check("disabled: sin reporte del gate", report(output) is None)
    check("disabled: sin cuarentena ni archivo de JSON",
          not os.path.exists(os.path.join(output, "dataset", "body_visibility"))
          and os.path.isfile(os.path.join(output, "dataset", "1_persons", "parquet_files",
                                          f"{CLIP}.parquet"))
          and not os.path.exists(os.path.join(os.path.dirname(json_dir(output)), "previous_json")),
          str(tools.argvs)[:120])
    check("disabled: sin --model_pose (comportamiento heredado)",
          all("--model_pose" not in a for a in tools.argvs))
    shutil.rmtree(root)


def test_disabled_does_not_read_video_metadata():
    """The gate must not add a cv2 dependency to the legacy path."""
    root, videos, output, frames = make_layout(4)
    tools = FakeTools(frames)
    real = argos.video_frame_count
    argos.video_frame_count = lambda video: (_ for _ in ()).throw(AssertionError("cv2 leído"))
    try:
        rc = argos.run(argos.build_parser().parse_args(
            ["--videos_folder", videos, "--output_folder", output]), runner=tools)
        check("disabled: no lee metadatos de video", rc == 0 and tools.r_runs == 1, str(rc))
    finally:
        argos.video_frame_count = real
    shutil.rmtree(root)


# ----------------------------------------------------- enabled accepted ----

def test_enabled_accepted():
    root, videos, output, frames = make_layout(10)
    tools = FakeTools(frames)
    rc = run_gate(videos, output, tools)[0]
    rep = report(output)
    check("enabled aceptado: exito + R + parquet",
          rc == 0 and tools.r_runs == 1 and os.path.isfile(
              os.path.join(output, "dataset", "1_persons", "parquet_files", f"{CLIP}.parquet")))
    check("enabled aceptado: reporte accepted con cobertura",
          rep and rep["decision"] == "accepted" and rep["expected_frames"] == frames
          and rep["points"][0]["coverage_percent"] == 100.0, str(rep)[:160])
    check("enabled: OpenPose forzado a BODY_25",
          any("--model_pose" in a and a[a.index("--model_pose") + 1] == "BODY_25" for a in tools.argvs),
          str(tools.argvs[:1])[:200])
    shutil.rmtree(root)


def test_enabled_custom_points_accepted():
    root, videos, output, frames = make_layout(10)
    # wrist 4 visible on frames 0-4, wrist 7 on frames 5-9: never together
    tools = FakeTools(frames, windows={4: (0, 5), 7: (5, 10)})
    rc = run_gate(videos, output, tools, **{"required-body-points": "4,7",
                                            "min-visible-percent": 50})[0]
    rep = report(output)
    check("enabled: ventanas disjoint aceptadas con --min-visible-percent 50",
          rc == 0 and rep["decision"] == "accepted"
          and [p["coverage_percent"] for p in rep["points"]] == [50.0, 50.0], str(rep)[:200])
    shutil.rmtree(root)


# ------------------------------------------------------------ rejections ----

def test_enabled_rejected_low_coverage():
    root, videos, output, frames = make_layout(10)
    write_parquet(os.path.join(output, "dataset"), 1, CLIP, tag="stale")
    tools = FakeTools(frames, windows={4: (0, 3), 7: (5, 10)})  # 30% / 50%
    rc = run_gate(videos, output, tools)[0]
    rep = report(output)
    ds = os.path.join(output, "dataset")
    check("baja cobertura: no se ejecuta R y sale 0", rc == 0 and tools.r_runs == 0)
    check("baja cobertura: reporte rejected", rep and rep["decision"] == "low_coverage"
          and rep["accepted"] is False, str(rep)[:160])
    check("baja cobertura: parquet rancio fuera de *_persons",
          not os.path.exists(os.path.join(ds, "1_persons", "parquet_files", f"{CLIP}.parquet"))
          and os.path.isfile(os.path.join(ds, "body_visibility", "rejected", "low_coverage",
                                          "parquet_files", f"{CLIP}__1_persons.parquet")))
    check("baja cobertura: conserva video y JSON",
          os.path.isfile(os.path.join(videos, f"{CLIP}.mp4"))
          and len([n for n in os.listdir(json_dir(output)) if n.endswith(bv.JSON_SUFFIX)]) == frames)
    shutil.rmtree(root)


def test_enabled_rejected_multi_person():
    root, videos, output, frames = make_layout(10)
    write_parquet(os.path.join(output, "dataset"), 1, CLIP, tag="stale")
    tools = FakeTools(frames, multi_person=True)
    rc = run_gate(videos, output, tools)[0]
    rep = report(output)
    ds = os.path.join(output, "dataset")
    check("multi-persona: sin R, reporte multi_person, salida 0",
          rc == 0 and tools.r_runs == 0 and rep["decision"] == "multi_person", str(rep)[:120])
    check("multi-persona: parquet cuarentenado en su categoria",
          os.path.isfile(os.path.join(ds, "body_visibility", "rejected", "multi_person",
                                      "parquet_files", f"{CLIP}__1_persons.parquet")))
    shutil.rmtree(root)


def test_rejection_does_not_touch_other_clips():
    root, videos, output, frames = make_layout(10)
    keep = write_parquet(os.path.join(output, "dataset"), 1, "another_clip", tag="keep")
    write_parquet(os.path.join(output, "dataset"), 2, CLIP, tag="stale2")
    tools = FakeTools(frames, windows={4: (0, 1), 7: (0, 1)})
    run_gate(videos, output, tools)
    ds = os.path.join(output, "dataset")
    check("otro clip conserva su parquet", os.path.isfile(keep))
    check("el mismo clip en otra categoria tambien se cuarentena",
          not os.path.exists(os.path.join(ds, "2_persons", "parquet_files", f"{CLIP}.parquet")))
    shutil.rmtree(root)


# --------------------------------------------------------------- errors ----

def test_openpose_error_is_fatal_for_the_run():
    root, videos, output, frames = make_layout(10)
    write_parquet(os.path.join(output, "dataset"), 1, CLIP, tag="stale")
    tools = FakeTools(frames, openpose_rc=1)
    rc = run_gate(videos, output, tools)[0]
    ds = os.path.join(output, "dataset")
    rep = report(output)
    check("error de OpenPose: salida distinta de 0 y sin R", rc != 0 and tools.r_runs == 0)
    check("error de OpenPose: no se mide JSON previo",
          not os.path.exists(json_dir(output)) or not os.listdir(json_dir(output)),
          str(os.listdir(json_dir(output))[:3]) if os.path.isdir(json_dir(output)) else "sin carpeta")
    check("error de OpenPose: reporte data_error + cuarentena",
          rep and rep["decision"] == "data_error" and rep["error"]
          and os.path.isfile(os.path.join(ds, "body_visibility", "rejected", "error",
                                          "parquet_files", f"{CLIP}__1_persons.parquet")), str(rep)[:200])
    shutil.rmtree(root)


def test_r_error_is_fatal_for_the_run():
    root, videos, output, frames = make_layout(10)
    tools = FakeTools(frames, r_rc=3)
    rc = run_gate(videos, output, tools)[0]
    check("error de R: salida distinta de 0 y reporte data_error",
          rc != 0 and report(output)["decision"] == "data_error", str(rc))
    shutil.rmtree(root)


def test_openpose_launch_error_is_reported_and_later_clips_run():
    """openpose.bin missing (FileNotFoundError) is a data_error, not a crash.

    The gate must still write its report, quarantine a stale Parquet, keep the
    clip's JSON, let the *next* clip be processed and exit non-zero.
    """
    root, videos, output, frames = make_layout(10)
    write_parquet(os.path.join(output, "dataset"), 1, CLIP, tag="stale")
    second = f"{CLIP}_second"
    actual2 = make_video(os.path.join(videos, f"{second}.mp4"), frames)
    tools = LaunchFailTools(frames, "openpose", FileNotFoundError, fail_once=True)
    rc = run_gate(videos, output, tools)[0]
    ds = os.path.join(output, "dataset")
    rep = report(output)
    check("launch OpenPose: salida 1, reporte data_error, sin R para ese clip",
          rc == 1 and rep and rep["decision"] == "data_error" and rep["stage"] == "openpose"
          and tools.r_runs == 1 and tools.seen.count("Rscript") == 1, f"rc={rc} rep={str(rep)[:160]}")
    check("launch OpenPose: parquet rancio cuarentenado",
          not os.path.exists(os.path.join(ds, "1_persons", "parquet_files", f"{CLIP}.parquet"))
          and os.path.isfile(os.path.join(ds, "body_visibility", "rejected", "error",
                                          "parquet_files", f"{CLIP}__1_persons.parquet")))
    rep2 = report_for(output, second)
    check("launch OpenPose: el clip siguiente se procesa",
          rep2 and rep2["decision"] == "accepted" and rep2["expected_frames"] == actual2,
          str(rep2)[:160])
    check("launch OpenPose: el clip afectado conserva su video",
          os.path.isfile(os.path.join(videos, f"{CLIP}.mp4")))
    shutil.rmtree(root)


def test_r_launch_error_overrides_the_accepted_report():
    """A non-executable Rscript must not leave the accepted verdict standing."""
    root, videos, output, frames = make_layout(10)
    write_parquet(os.path.join(output, "dataset"), 1, CLIP, tag="stale")
    tools = LaunchFailTools(frames, "Rscript", PermissionError)
    rc = run_gate(videos, output, tools)[0]
    ds = os.path.join(output, "dataset")
    rep = report(output)
    check("launch R: salida 1 y reporte data_error que sobrescribe el accepted",
          rc == 1 and rep and rep["decision"] == "data_error" and rep["accepted"] is False
          and rep["stage"] == "df_maker" and "max_people_classification.R" in rep["error"],
          str(rep)[:200])
    check("launch R: sin parquet y JSON de esta pasada conservado",
          not os.path.exists(os.path.join(ds, "1_persons", "parquet_files", f"{CLIP}.parquet"))
          and os.path.isfile(os.path.join(ds, "body_visibility", "rejected", "error",
                                          "parquet_files", f"{CLIP}__1_persons.parquet"))
          and len([n for n in os.listdir(json_dir(output)) if n.endswith(bv.JSON_SUFFIX)]) == frames,
          str(os.listdir(json_dir(output))[:2]) if os.path.isdir(json_dir(output)) else "sin JSON")
    shutil.rmtree(root)


def test_launch_error_still_reports_when_archive_fails():
    """An unusable previous JSON folder must not skip the report or the quarantine."""
    root, videos, output, frames = make_layout(10)
    write_parquet(os.path.join(output, "dataset"), 1, CLIP, tag="stale")
    tools = LaunchFailTools(frames, "openpose", FileNotFoundError)
    real_archive = bv.archive_json_dir
    bv.archive_json_dir = lambda *a, **kw: (_ for _ in ()).throw(
        OSError("archive is read-only"))
    try:
        rc = run_gate(videos, output, tools)[0]
    finally:
        bv.archive_json_dir = real_archive
    ds = os.path.join(output, "dataset")
    rep = report(output)
    check("fallo al archivar: reporte data_error + cuarentena igualmente",
          rc == 1 and rep and rep["decision"] == "data_error"
          and os.path.isfile(os.path.join(ds, "body_visibility", "rejected", "error",
                                          "parquet_files", f"{CLIP}__1_persons.parquet")),
          f"rc={rc} rep={str(rep)[:160]}")
    shutil.rmtree(root)


def test_unexpected_error_during_archive_is_not_swallowed():
    """Only expected filesystem errors are routed; a bug must still escape."""
    root, videos, output, frames = make_layout(10)
    tools = FakeTools(frames)
    real_archive = bv.archive_json_dir
    bv.archive_json_dir = lambda *a, **kw: (_ for _ in ()).throw(ZeroDivisionError("bug"))
    try:
        try:
            run_gate(videos, output, tools)
            raised = None
        except BaseException as exc:  # noqa: BLE001
            raised = exc
    finally:
        bv.archive_json_dir = real_archive
    check("fallo inesperado al archivar: no se traga", isinstance(raised, ZeroDivisionError),
          repr(raised))
    shutil.rmtree(root)


def test_malformed_openpose_json_is_quarantined_via_cli():
    """Unparsable OpenPose output reaches the CLI as a fail-closed data_error."""
    root, videos, output, frames = make_layout(10)
    write_parquet(os.path.join(output, "dataset"), 1, CLIP, tag="stale")
    real_writer = write_openpose_json

    class BrokenJSONTools(FakeTools):
        def __call__(self, argv, cwd=None, quiet=False):
            if argv[0].endswith("openpose.bin"):
                json_dir_ = argv[argv.index("--write_json") + 1]
                real_writer(json_dir_, CLIP, self.frames, self.windows, self.conf)
                with open(os.path.join(json_dir_, f"{CLIP}_{3:012d}_keypoints.json"), "wb") as fh:
                    fh.write(b'{"people": [{"pose_keypoints_2d": [\xff\xfe\x00')
                self.argvs.append(list(argv))
                return self.openpose_rc
            return super().__call__(argv, cwd=cwd, quiet=quiet)

    tools = BrokenJSONTools(frames)
    rc = run_gate(videos, output, tools)[0]
    ds = os.path.join(output, "dataset")
    rep = report(output)
    check("JSON malformado via CLI: data_error, sin R, cuarentena",
          rc == 1 and tools.r_runs == 0 and rep and rep["decision"] == "data_error"
          and os.path.isfile(os.path.join(ds, "body_visibility", "rejected", "error",
                                          "parquet_files", f"{CLIP}__1_persons.parquet")),
          f"rc={rc} rep={str(rep)[:160]}")
    check("JSON malformado via CLI: el JSON se conserva archivado",
          not os.path.exists(json_dir(output))
          or not [n for n in os.listdir(json_dir(output)) if n.endswith(bv.JSON_SUFFIX)])
    archive = os.path.join(os.path.dirname(json_dir(output)), "previous_json")
    check("JSON malformado via CLI: nada se borra",
          os.path.isdir(archive)
          and len([n for n in os.listdir(archive) if n.endswith(bv.JSON_SUFFIX)]) == frames,
          str(sorted(os.listdir(archive))[:2]) if os.path.isdir(archive) else "sin archivo")
    shutil.rmtree(root)


def test_huge_json_integer_is_quarantined_via_cli():
    """A 5001-digit number is valid JSON that CPython refuses to parse.

    `json.load` raises a plain ValueError there, which used to escape the gate as
    a traceback: no report, no quarantine, and no later clips. It must behave
    like any other data error.
    """
    root, videos, output, frames = make_layout(10)
    write_parquet(os.path.join(output, "dataset"), 1, CLIP, tag="stale")
    second = f"{CLIP}_second"
    make_video(os.path.join(videos, f"{second}.mp4"), frames)
    huge = "1" + "0" * 5000
    real_writer = write_openpose_json

    def values(idx):
        out = ["0.0"] * 75
        out[3 * 4], out[3 * 4 + 1], out[3 * 4 + 2] = huge if idx == 4 else "0.0", "1.0", "0.9"
        return out

    class HugeIntTools(FakeTools):
        def __call__(self, argv, cwd=None, quiet=False):
            if argv[0].endswith("openpose.bin"):
                json_dir_ = argv[argv.index("--write_json") + 1]
                if os.path.basename(os.path.dirname(json_dir_)) == second:
                    return super().__call__(argv, cwd=cwd, quiet=quiet)
                os.makedirs(json_dir_, exist_ok=True)
                for idx in range(self.frames):
                    # written as text: json.dump of such an int raises on its own
                    with open(os.path.join(json_dir_, f"{CLIP}_{idx:012d}_keypoints.json"), "w") as fh:
                        fh.write('{"version": "1.5.1", "people": [{"pose_keypoints_2d": ['
                                 + ", ".join(values(idx)) + ']}]}')
                self.argvs.append(list(argv))
                return self.openpose_rc
            return super().__call__(argv, cwd=cwd, quiet=quiet)

    tools = HugeIntTools(frames)
    rc = run_gate(videos, output, tools)[0]
    ds = os.path.join(output, "dataset")
    rep = report(output)
    check("entero enorme via CLI: data_error controlado, sin R para ese clip, cuarentena",
          rc == 1 and tools.r_runs == 1 and rep and rep["decision"] == "data_error"
          and not os.path.exists(os.path.join(ds, "1_persons", "parquet_files",
                                              f"{CLIP}.parquet"))
          and os.path.isfile(os.path.join(ds, "body_visibility", "rejected", "error",
                                          "parquet_files", f"{CLIP}__1_persons.parquet")),
          f"rc={rc} r_runs={tools.r_runs} rep={str(rep)[:160]}")
    check("entero enorme via CLI: los JSON presentes no se declaran faltantes",
          rep["frames_with_json"] == frames and rep["frames_missing_json"] == 0
          and rep["statistics_complete"] is False, str(rep)[:240])
    check("entero enorme via CLI: el clip siguiente se procesa",
          report_for(output, second)["decision"] == "accepted",
          str(report_for(output, second))[:160])
    shutil.rmtree(root)


def test_unusable_video_metadata_is_fatal():
    root, videos, output, frames = make_layout(10)
    tools = FakeTools(frames)
    real = argos.video_frame_count
    argos.video_frame_count = lambda video: 0
    try:
        rc = run_gate(videos, output, tools)[0]
        check("sin metadatos de video: falla cerrado sin ejecutar nada",
              rc != 0 and tools.argvs == [] and report(output)["decision"] == "data_error", str(rc))
    finally:
        argos.video_frame_count = real
    shutil.rmtree(root)


def test_broken_video_metadata_is_a_controlled_data_error():
    """A frame count that cannot be read is reported, not raised through the CLI.

    OpenPose/ffprobe can hand back a file whose container metadata is unusable:
    NaN, inf, a 401-digit count or a decoder that throws. None of those may
    escape `polymnia argos` as a traceback, because that drops the report,
    the quarantine and every later clip.
    """
    cases = {
        "nan": lambda video: float("nan"),
        "inf": lambda video: float("inf"),
        "entero enorme": lambda video: 10 ** 400,
        "lanzador de error": lambda video: (_ for _ in ()).throw(
            bv.OpenPoseDataError("decoder exploded")),
        "tipo raro": lambda video: "once",
    }
    for name, counter in cases.items():
        root, videos, output, frames = make_layout(10)
        write_parquet(os.path.join(output, "dataset"), 1, CLIP, tag="stale")
        tools = FakeTools(frames)
        real = argos.video_frame_count
        argos.video_frame_count = counter
        try:
            try:
                rc = run_gate(videos, output, tools)[0]
                raised = None
            except BaseException as exc:  # noqa: BLE001
                rc, raised = -999, exc
        finally:
            argos.video_frame_count = real
        rep = report(output)
        ds = os.path.join(output, "dataset")
        check(f"metadatos de video ({name}): data_error controlado sin traza",
              raised is None and rc == 1 and rep and rep["decision"] == "data_error"
              and rep["stage"] == "video_metadata" and tools.argvs == []
              and os.path.isfile(os.path.join(ds, "body_visibility", "rejected", "error",
                                              "parquet_files", f"{CLIP}__1_persons.parquet")),
              f"raised={raised!r} rc={rc} rep={str(rep)[:120]}")
        shutil.rmtree(root)


def test_broken_metadata_of_one_clip_does_not_stop_later_clips():
    root, videos, output, frames = make_layout(10)
    second = f"{CLIP}_second"
    make_video(os.path.join(videos, f"{second}.mp4"), frames)
    real = argos.video_frame_count

    def broken(video):
        if os.path.basename(video).endswith("_second.mp4"):
            return real(video)
        raise RuntimeError("decoder crashed")

    argos.video_frame_count = broken
    try:
        rc = run_gate(videos, output, FakeTools(frames))[0]
    finally:
        argos.video_frame_count = real
    check("metadatos rotos: el clip siguiente igual se mide",
          rc == 1 and report(output)["decision"] == "data_error"
          and report_for(output, second)["decision"] == "accepted", str(rc))
    shutil.rmtree(root)


# --------------------------------------------------- stale JSON handling ----

def test_enabled_rerun_archives_previous_json():
    root, videos, output, frames = make_layout(6)
    os.makedirs(json_dir(output))
    write_openpose_json(json_dir(output), "other_clip", 6)  # leftover from an old run
    leftover = os.path.join(json_dir(output), "num_persons.txt")
    with open(leftover, "w") as fh:
        fh.write("2\n")
    tools = FakeTools(frames, windows={4: (0, 2), 7: (3, 5)})  # 33% / 33% -> rejected
    rc = run_gate(videos, output, tools)[0]
    archive = os.path.join(os.path.dirname(json_dir(output)), "previous_json")
    archived = sorted(os.listdir(archive)) if os.path.isdir(archive) else []
    expected_archived = sorted(["num_persons.txt"] + [
        f"other_clip_{i:012d}_keypoints.json" for i in range(6)])
    check("rerun: el JSON anterior se conserva aparte", archived == expected_archived, str(archived[:3]))
    check("rerun: se mide solo el JSON de esta pasada",
          rc == 0 and report(output)["decision"] == "low_coverage"
          and report(output)["frames_no_person"] == 0, str(report(output))[:200])
    shutil.rmtree(root)


def test_disabled_rerun_keeps_previous_json_in_place():
    root, videos, output, frames = make_layout(5)
    os.makedirs(json_dir(output))
    write_openpose_json(json_dir(output), "other_clip", 5)
    tools = FakeTools(frames)
    argos.run(argos.build_parser().parse_args(
        ["--videos_folder", videos, "--output_folder", output]), runner=tools)
    check("disabled: no reubica el JSON existente",
          not os.path.exists(os.path.join(os.path.dirname(json_dir(output)), "previous_json"))
          and len([n for n in os.listdir(json_dir(output)) if n.endswith(bv.JSON_SUFFIX)]) == 10,
          str(sorted(os.listdir(json_dir(output)))[:2]))
    shutil.rmtree(root)


# ----------------------------------------------------------- CLI parsing ----

def test_cli_defaults_and_validation():
    p = argos.build_parser()
    base = ["--videos_folder", "v", "--output_folder", "o"]
    unwanted = lambda argv, cwd=None, quiet=False: (_ for _ in ()).throw(AssertionError("ejecuto algo"))  # noqa: E731
    plain = p.parse_args(base)
    check("gate desactivado por defecto", getattr(plain, "filter_body_visibility") is False)
    on = p.parse_args(base + ["--filter-body-visibility"])
    check("defaults 4,7 / 40 / 0.2",
          getattr(on, "required_body_points") == "4,7"
          and float(getattr(on, "min_visible_percent")) == 40.0
          and float(getattr(on, "min_keypoint_confidence")) == 0.2)
    rc = argos.run(p.parse_args(base + ["--filter-body-visibility", "--min-visible-percent", "140"]),
                   runner=unwanted)
    check("config invalida: sale distinto de 0 sin ejecutar nada", rc != 0, str(rc))
    rc = argos.run(p.parse_args(base + ["--videos_folder", "/nonexistent-folder-for-test"]),
                   runner=unwanted)


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[fn]()
        except Exception as e:  # noqa: BLE001
            check(f"{fn} (excepcion)", False, repr(e))
    print("\nTOTAL:", "FAIL " + ", ".join(FAIL) if FAIL else "todo OK")
    sys.exit(1 if FAIL else 0)
