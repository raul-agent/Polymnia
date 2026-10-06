"""Tests for body_visibility.py, the optional BODY_25 clip gate.

The contract under test: every required BODY_25 point must be confidently
visible in at least min_visible_percent of ALL frames of the source video
(the denominator is the video's frame count, not the number of JSON files),
and the windows where each point is visible do not have to coincide.

Everything here is synthetic JSON on a tmpdir: no GPU, no OpenPose, no R.
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import body_visibility as bv  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------ fixtures ----

CLIP = "clip_one"


def frame(index, persons):
    """One OpenPose JSON frame: [{pose_keypoints_2d: [...]}, ...]."""
    return {"version": "1.5.1", "people": [{"pose_keypoints_2d": p} for p in persons]}


def body(points):
    """BODY_25 array from {index: (x, y, conf)}; everything else is absent."""
    arr = [0.0] * 75
    for i, (x, y, c) in points.items():
        arr[3 * i:3 * i + 3] = [x, y, c]
    return arr


def write_json(json_dir, index, payload):
    os.makedirs(json_dir, exist_ok=True)
    path = os.path.join(json_dir, f"{CLIP}_{index:012d}_keypoints.json")
    with open(path, "w") as fh:
        if isinstance(payload, str):
            fh.write(payload)
        else:
            json.dump(payload, fh)
    return path


def build_json_dir(expected_frames, visible, json_dir=None, conf=0.5, points=(4, 7), overlap=False):
    """Frames where each required point is visible `visible[point]` times.

    By default the windows are disjoint (point 4 first, point 7 after it), which
    is what the gate must tolerate; `overlap=True` stacks them from frame 0.
    """
    json_dir = json_dir or make_json_dir()
    windows = {}
    start = 0
    for p in points:
        windows[p] = range(0, visible[p]) if overlap else range(start, start + visible[p])
        start += visible[p]
    for idx in range(expected_frames):
        shown = {p: (float(idx), 1.0, conf) for p in points if idx in windows[p]}
        write_json(json_dir, idx, frame(idx, [body(shown)] if shown else []))
    return json_dir


def make_json_dir():
    tmp = tempfile.mkdtemp(prefix="bv-")
    return os.path.join(tmp, "JSON_FILES")


def cfg(points=(4, 7), percent=40.0, conf=0.2):
    return bv.VisibilityConfig.from_args(points, percent, conf)


def one_decision(visible, expected=10, **kw):
    points = tuple(kw.pop("points", (4, 7)))
    jd = build_json_dir(expected, {4: visible[0], 7: visible[1]}, points=points, **kw)
    return bv.evaluate_body_visibility(jd, expected, cfg(points=points))


# ------------------------------------------------------------- coverage ----

def test_disjoint_windows_accepted():
    d = one_decision((4, 6))  # 4 -> 40%, 7 -> 60%, never together
    check("ventanas disjoint -> accept (40%/60% de 10)", d.accepted and d.reason == bv.REASON_ACCEPTED,
          str(d.as_dict()["points"]))


def test_denominator_is_all_video_frames():
    jd = make_json_dir()
    for idx in range(4):  # only 4 JSON files, visible in all of them
        write_json(jd, idx, frame(idx, [body({4: (1, 1, 0.9), 7: (2, 2, 0.9)})]))
    d = bv.evaluate_body_visibility(jd, 10, cfg())
    p4 = [p for p in d.points if p["point"] == 4][0]
    check("faltan JSON -> cuentan ausentes (4/10 = 40%)",
          d.accepted and p4["visible_frames"] == 4 and p4["coverage_percent"] == 40.0
          and d.frames_missing_json == 6, str(d.as_dict()))


def test_exact_threshold_boundary():
    ok = one_decision((4, 4)).accepted           # 40.0% >= 40%
    bad = one_decision((3, 9)).accepted          # 30% < 40%
    check("exactamente 40% pasa", ok)
    check("39.9% (3/10) rechaza", not bad)


def test_frames_without_person_count_absent():
    d = one_decision((0, 9))  # point 4 never visible
    low = [p for p in d.points if p["point"] == 4][0]
    check("punto nunca visible -> reject low_coverage",
          not d.accepted and d.reason == bv.REASON_LOW_COVERAGE and low["visible_frames"] == 0,
          d.reason)


def test_confidence_threshold_inclusive():
    jd = build_json_dir(10, {4: 10, 7: 10}, conf=0.2, overlap=True)
    check("conf == umbral cuenta visible", bv.evaluate_body_visibility(jd, 10, cfg()).accepted)
    jd = build_json_dir(10, {4: 10, 7: 10}, conf=0.199, overlap=True)
    check("conf < umbral cuenta ausente", not bv.evaluate_body_visibility(jd, 10, cfg()).accepted)


def test_zero_conf_absent_even_at_zero_threshold():
    jd = build_json_dir(10, {4: 10, 7: 10}, conf=0.0, overlap=True)
    d = bv.evaluate_body_visibility(jd, 10, cfg(conf=0.0))
    check("c=0 es ausente aunque el umbral sea 0", not d.accepted, str(d.as_dict()["points"]))
    jd = build_json_dir(10, {4: 10, 7: 10}, conf=1e-9, overlap=True)
    check("con umbral 0, c=1e-9 cuenta visible", bv.evaluate_body_visibility(jd, 10, cfg(conf=0.0)).accepted)


def test_custom_points_and_report():
    jd = build_json_dir(10, {2: 5, 24: 5}, points=(2, 24))
    d = bv.evaluate_body_visibility(jd, 10, cfg(points=(2, 24)))
    got = {p["point"]: p["coverage_percent"] for p in d.points}
    check("puntos configurables 2,24 en ventanas disjoint", d.accepted and got == {2: 50.0, 24: 50.0}, str(got))
    rep = d.as_dict()
    check("reporte lleva denominador y config",
          rep["expected_frames"] == 10 and rep["decision"] == "accepted"
          and rep["config"]["required_points"] == [2, 24]
          and rep["config"]["min_visible_percent"] == 40.0
          and rep["config"]["min_keypoint_confidence"] == 0.2, str(rep)[:200])


# ---------------------------------------------------------- rejections ----

def test_multi_person_rejects():
    jd = make_json_dir()
    for idx in range(10):
        ppl = [body({4: (1, 1, 0.9), 7: (2, 2, 0.9)})]
        if idx == 3:
            ppl.append(body({4: (5, 5, 0.9), 7: (6, 6, 0.9)}))
        write_json(jd, idx, frame(idx, ppl))
    d = bv.evaluate_body_visibility(jd, 10, cfg())
    check("un frame con 2 personas rechaza el clip",
          not d.accepted and d.reason == bv.REASON_MULTI_PERSON and not d.error, d.reason)


def test_multi_person_frames_are_not_missing_json():
    """A multi-person frame has JSON: it must count as present, not as missing.

    The report is the only thing a reviewer sees, so `frames_with_json` must be
    the real number of parsed frames and the multi-person frames must show up
    under their own heading, never folded into `frames_missing_json`.
    """
    jd = make_json_dir()
    for idx in range(10):
        ppl = [body({4: (1, 1, 0.9), 7: (2, 2, 0.9)})]
        if idx in (3, 7):
            ppl.append(body({4: (5, 5, 0.9), 7: (6, 6, 0.9)}))
        write_json(jd, idx, frame(idx, ppl))
    rep = bv.evaluate_body_visibility(jd, 10, cfg()).as_dict()
    check("multi-persona: JSON contado como presente, no como ausente",
          rep["decision"] == bv.REASON_MULTI_PERSON and rep["frames_with_json"] == 10
          and rep["frames_missing_json"] == 0 and rep["frames_multi_person"] == 2
          and rep["expected_frames"] == 10, str(rep)[:220])


def test_missing_and_multi_person_are_reported_apart():
    """10 frames expected, 8 with JSON (1 of them multi-person), 2 without."""
    jd = make_json_dir()
    for idx in range(8):
        ppl = [body({4: (1, 1, 0.9), 7: (2, 2, 0.9)})]
        if idx == 4:
            ppl.append(body({4: (5, 5, 0.9), 7: (6, 6, 0.9)}))
        write_json(jd, idx, frame(idx, ppl))
    rep = bv.evaluate_body_visibility(jd, 10, cfg()).as_dict()
    check("faltantes y multi-persona se reportan por separado",
          rep["frames_with_json"] == 8 and rep["frames_missing_json"] == 2
          and rep["frames_multi_person"] == 1, str(rep)[:220])


# ----------------------------------------------------- fail-closed data ----

def good_payload(idx):
    return frame(idx, [body({4: (float(idx), 1.0, 0.9), 7: (2.0, 2.0, 0.9)})])


def test_fail_closed_on_bad_data():
    cases = {
        "json invalido": lambda idx: "{not json" if idx == 2 else json.dumps(good_payload(idx)),
        "float no finita": lambda idx: json.dumps(
            {"version": "1.5.1", "people": [{"pose_keypoints_2d":
                [float("nan")] + good_payload(idx)["people"][0]["pose_keypoints_2d"][1:]}]}),
        "skeleton BODY_25 incorrecto": lambda idx: json.dumps(
            {"version": "1.5.1", "people": [{"pose_keypoints_2d": [0.0] * 72}]}),
        "falta pose_keypoints_2d": lambda idx: json.dumps({"version": "1.5.1", "people": [{}]}),
    }
    for name, payload_for in cases.items():
        jd = make_json_dir()
        for idx in range(10):
            write_json(jd, idx, payload_for(idx))
        try:
            d = bv.evaluate_body_visibility(jd, 10, cfg())
            ok, detail = bool(d.error), f"error={d.error} reason={d.reason}"
        except bv.OpenPoseDataError as e:
            ok, detail = True, f"raised {e}"
        check(f"fail closed: {name}", ok, detail)

    # duplicated frame indices (two names claiming the same frame) are detected
    jd = make_json_dir()
    for idx in range(10):
        write_json(jd, idx, good_payload(idx))
    write_json(jd, 4, good_payload(4))
    with open(os.path.join(jd, "other_clip_000000000006" + bv.JSON_SUFFIX), "w") as fh:
        json.dump(good_payload(6), fh)
    try:
        d = bv.evaluate_body_visibility(jd, 10, cfg())
        check("fail closed: indice duplicado", bool(d.error), str(d.as_dict())[:120])
    except bv.OpenPoseDataError as e:
        check("fail closed: indice duplicado", True, f"raised {e}")

    jd = make_json_dir()
    write_json(jd, 10, good_payload(10))
    try:
        d = bv.evaluate_body_visibility(jd, 10, cfg())
        check("fail closed: indice fuera de rango", bool(d.error), str(d.as_dict())[:120])
    except bv.OpenPoseDataError as e:
        check("fail closed: indice fuera de rango", True, f"raised {e}")


ORIGINAL_INT_LIMIT = sys.get_int_max_str_digits()

HUGE_DIGITS = "1" + "0" * 5000  # 5001 digits: legal JSON, past CPython's 4300-digit int limit


def huge_int_json():
    """Valid JSON text with a 5001-digit number inside pose_keypoints_2d.

    Written by hand on purpose: `json.dump` of a Python int that large raises
    before the fixture ever reaches the API under test.
    """
    values = ["0.0"] * 75
    values[3 * 4] = HUGE_DIGITS
    return ('{"version": "1.5.1", "people": [{"pose_keypoints_2d": ['
            + ", ".join(values) + "]}]}")


def test_fail_closed_on_huge_json_integer():
    """Legal JSON that CPython refuses to turn into an int is a data error.

    `json.load` raises a *plain* ValueError (not JSONDecodeError) for integers
    with more digits than the int->str limit, so the known parser ValueError has
    to become an OpenPoseDataError instead of a traceback escaping the gate.
    The global digit limit is deliberately left as it is.
    """
    jd = make_json_dir()
    for idx in range(10):
        write_json(jd, idx, huge_int_json() if idx == 2 else json.dumps(good_payload(idx)))
    try:
        d = bv.evaluate_body_visibility(jd, 10, cfg())
        check("fail closed: entero JSON de 5001 digitos",
              d.reason == bv.REASON_DATA_ERROR and bool(d.error)
              and d.frames_with_json == 10 and d.frames_missing_json == 0,
              f"reason={d.reason} error={d.error}")
    except bv.OpenPoseDataError as e:
        check("fail closed: entero JSON de 5001 digitos", True, f"raised {e}")
    check("no se sube el limite global de digitos de enteros",
          sys.get_int_max_str_digits() == ORIGINAL_INT_LIMIT,
          str(sys.get_int_max_str_digits()))


# ------------------------------------------------- statistics of a partial ----

def test_malformed_json_in_middle_reports_files_that_exist():
    """10 frames expected, 10 JSON files present, frame 5 unparsable.

    Counting stops at the bad frame, but the other nine files still exist: the
    presence statistics come from the folder inventory, so a report must never
    turn files that are there into "missing JSON".
    """
    jd = make_json_dir()
    for idx in range(10):
        write_json(jd, idx, "{not json" if idx == 5 else json.dumps(good_payload(idx)))
    rep = bv.evaluate_body_visibility(jd, 10, cfg()).as_dict()
    check("JSON malformado en el medio: los archivos presentes no se declaran faltantes",
          rep["decision"] == bv.REASON_DATA_ERROR and rep["frames_with_json"] == 10
          and rep["frames_missing_json"] == 0 and rep["frames_processed"] == 5
          and rep["statistics_complete"] is False, str(rep)[:240])


def test_bad_skeleton_in_middle_keeps_real_missing_count():
    """5 frames expected, only 3 JSON files, the middle one has a wrong skeleton.

    `frames_with_json` is the inventory (3), so the missing count is the genuine
    two files that were never written, not the five frames counting never reached.
    """
    jd = make_json_dir()
    for idx in range(3):
        payload = good_payload(idx)
        if idx == 1:
            payload["people"][0]["pose_keypoints_2d"] = [0.0] * 72
        write_json(jd, idx, payload)
    rep = bv.evaluate_body_visibility(jd, 5, cfg()).as_dict()
    check("skeleton roto en el medio: presente = inventario, faltantes = los reales",
          rep["decision"] == bv.REASON_DATA_ERROR and rep["frames_with_json"] == 3
          and rep["frames_missing_json"] == 2 and rep["frames_processed"] == 1
          and rep["statistics_complete"] is False, str(rep)[:240])


def test_no_person_stats_stopped_at_error_are_partial():
    """Frames counted before the bad one stay in the report, marked as partial."""
    jd = make_json_dir()
    for idx in range(6):
        if idx == 5:
            write_json(jd, idx, "{not json")
        elif idx in (2, 3):
            write_json(jd, idx, frame(idx, []))
        else:
            write_json(jd, idx, good_payload(idx))
    rep = bv.evaluate_body_visibility(jd, 10, cfg()).as_dict()
    check("estadisticas parciales: statistics_complete=false con frames_processed",
          rep["frames_with_json"] == 6 and rep["frames_missing_json"] == 4
          and rep["frames_no_person"] == 2 and rep["frames_processed"] == 5
          and rep["statistics_complete"] is False, str(rep)[:240])


def test_multi_person_stats_stopped_at_error_are_partial():
    jd = make_json_dir()
    for idx in range(8):
        if idx == 7:
            write_json(jd, idx, json.dumps({"version": "1.5.1",
                                            "people": [{"pose_keypoints_2d": [0.0] * 72}]}))
            continue
        ppl = [body({4: (1, 1, 0.9), 7: (2, 2, 0.9)})]
        if idx == 1:
            ppl.append(body({4: (5, 5, 0.9), 7: (6, 6, 0.9)}))
        write_json(jd, idx, frame(idx, ppl))
    rep = bv.evaluate_body_visibility(jd, 10, cfg()).as_dict()
    check("multi-persona interrumpido por un error: parcial y con inventario completo",
          rep["decision"] == bv.REASON_DATA_ERROR and rep["frames_multi_person"] == 1
          and rep["frames_with_json"] == 8 and rep["frames_missing_json"] == 2
          and rep["statistics_complete"] is False, str(rep)[:240])


def test_inventory_failure_leaves_presence_unknown():
    """When the file names themselves are untrustworthy, nothing is claimed.

    A duplicated index, an index beyond the video, an unexpected name or a
    missing folder mean the inventory never finished: presence and absence are
    reported as unknown (null), never as a made-up number of missing files.
    """
    def duplicated():
        jd = make_json_dir()
        for idx in range(5):
            write_json(jd, idx, good_payload(idx))
        with open(os.path.join(jd, f"other_{4:012d}_keypoints.json"), "w") as fh:
            json.dump(good_payload(4), fh)
        return jd

    def out_of_range():
        jd = make_json_dir()
        for idx in range(3):
            write_json(jd, idx, good_payload(idx))
        write_json(jd, 12, good_payload(12))
        return jd

    def unexpected_name():
        jd = make_json_dir()
        write_json(jd, 0, good_payload(0))
        with open(os.path.join(jd, "summary" + bv.JSON_SUFFIX), "w") as fh:
            json.dump(good_payload(0), fh)
        return jd

    def absent_dir():
        return os.path.join(tempfile.mkdtemp(prefix="bv-none-"), "JSON_FILES")

    for name, build in (("indice duplicado", duplicated), ("indice fuera de rango", out_of_range),
                        ("nombre inesperado", unexpected_name), ("carpeta ausente", absent_dir)):
        try:
            rep = bv.evaluate_body_visibility(build(), 10, cfg()).as_dict()
            ok = (rep["decision"] == bv.REASON_DATA_ERROR and rep["frames_with_json"] is None
                  and rep["frames_missing_json"] is None
                  and rep["statistics_complete"] is False)
            detail = str(rep)[:200]
        except bv.OpenPoseDataError as e:
            ok, detail = False, f"raised {e}"
        check(f"inventario invalido ({name}): presencia y ausencia desconocidas", ok, detail)


def test_successful_report_statistics_are_complete():
    """The successful report keeps its meaning: complete statistics, no partial keys."""
    rep = one_decision((5, 5)).as_dict()
    check("reporte accepted: estadisticas completas y sin claves parciales",
          rep["decision"] == bv.REASON_ACCEPTED and rep["statistics_complete"] is True
          and "frames_processed" not in rep and rep["frames_with_json"] == 10
          and rep["frames_missing_json"] == 0, str(rep)[:240])
    jd = make_json_dir()
    for idx in range(10):
        ppl = [body({4: (1, 1, 0.9), 7: (2, 2, 0.9)})]
        if idx == 3:
            ppl.append(body({4: (5, 5, 0.9), 7: (6, 6, 0.9)}))
        write_json(jd, idx, frame(idx, ppl))
    rep = bv.evaluate_body_visibility(jd, 10, cfg()).as_dict()
    check("reporte multi_person: estadisticas completas", rep["statistics_complete"] is True
          and "frames_processed" not in rep and rep["frames_with_json"] == 10, str(rep)[:200])


def test_fail_closed_on_untrustworthy_scores():
    """Scores must be genuine JSON numbers inside [0, 1], for every BODY_25 point.

    `float()` alone accepts `true`, `"0.9"`, a 401-digit integer (OverflowError)
    and a score of 2 as if they were measured confidence, which turns garbage
    output into a coverage judgement. All of it is a data error instead.
    """
    cases = {
        "confianza 2 fuera de rango": {4: (1.0, 1.0, 2.0)},
        "confianza negativa": {4: (1.0, 1.0, -0.5)},
        "confianza booleana true": {4: (1.0, 1.0, True)},
        "confianza booleana false": {4: (1.0, 1.0, False)},
        "confianza string '0.9'": {4: (1.0, 1.0, "0.9")},
        "coordenada string": {4: ("1.0", 1.0, 0.9)},
        "entero de 401 digitos": {4: (10 ** 400, 1.0, 0.9)},
        "confianza entera de 401 digitos": {4: (1.0, 1.0, 10 ** 400)},
        "confianza inf": {4: (1.0, 1.0, float("inf"))},
    }
    for name, broken in cases.items():
        jd = make_json_dir()
        for idx in range(10):
            payload = good_payload(idx)
            if idx == 2:
                payload["people"][0]["pose_keypoints_2d"][3 * 4:3 * 4 + 3] = list(broken[4])
            write_json(jd, idx, payload)
        try:
            d = bv.evaluate_body_visibility(jd, 10, cfg())
            ok = d.reason == bv.REASON_DATA_ERROR and bool(d.error)
            detail = f"reason={d.reason} error={d.error}"
        except bv.OpenPoseDataError as e:
            ok, detail = True, f"raised {e}"
        check(f"fail closed: {name}", ok, detail)


def test_fail_closed_on_bad_confidence_of_an_optional_point():
    """The [0, 1] range applies to all 25 triples, not only the required points."""
    jd = make_json_dir()
    for idx in range(10):
        payload = good_payload(idx)
        if idx == 2:
            payload["people"][0]["pose_keypoints_2d"][3 * 0:3 * 0 + 3] = [1.0, 1.0, 1.5]
        write_json(jd, idx, payload)
    try:
        d = bv.evaluate_body_visibility(jd, 10, cfg())
        check("fail closed: confianza invalida en un punto no pedido",
              d.reason == bv.REASON_DATA_ERROR, str(d.as_dict())[:160])
    except bv.OpenPoseDataError as e:
        check("fail closed: confianza invalida en un punto no pedido", True, f"raised {e}")


def test_scores_at_the_range_edges_still_measure():
    """0.0 means \"not detected\" and 1.0 means certainty: both are legal data."""
    jd = build_json_dir(10, {4: 5, 7: 5}, conf=1.0, overlap=True)
    check("confianza 1.0 sigue siendo medible", bv.evaluate_body_visibility(jd, 10, cfg()).accepted)
    jd = make_json_dir()
    for idx in range(10):
        write_json(jd, idx, frame(idx, [body({4: (float(idx), 1.0, 0.0),
                                             7: (2.0, 2.0, 1.0), 0: (0.5, 0.5, 0.0)})]))
    d = bv.evaluate_body_visibility(jd, 10, cfg())
    check("confianza 0.0 en puntos ausentes no es error de datos",
          d.reason == bv.REASON_LOW_COVERAGE and not d.error, str(d.as_dict())[:160])


def test_fail_closed_on_invalid_utf8():
    """A truncated/binary file is unparsable data, not a UnicodeDecodeError crash."""
    jd = make_json_dir()
    for idx in range(10):
        write_json(jd, idx, good_payload(idx))
    with open(os.path.join(jd, f"{CLIP}_{5:012d}_keypoints.json"), "wb") as fh:
        fh.write(b'{"version": "1.5.1", "people": [{"pose_keypoints_2d": [\xff\xfe\x00')
    try:
        d = bv.evaluate_body_visibility(jd, 10, cfg())
        check("fail closed: UTF-8 invalido", d.reason == bv.REASON_DATA_ERROR and bool(d.error),
              f"reason={d.reason} error={d.error}")
    except bv.OpenPoseDataError as e:
        check("fail closed: UTF-8 invalido", True, f"raised {e}")

    with open(os.path.join(jd, f"{CLIP}_{7:012d}_keypoints.json"), "wb") as fh:
        fh.write(b"\x00\x01\x02binary")
    try:
        d = bv.evaluate_body_visibility(jd, 10, cfg())
        check("fail closed: archivo binario", d.reason == bv.REASON_DATA_ERROR and bool(d.error),
              f"reason={d.reason} error={d.error}")
    except bv.OpenPoseDataError as e:
        check("fail closed: archivo binario", True, f"raised {e}")


def test_report_written_atomically():
    tmp = tempfile.mkdtemp(prefix="bv-rep-")
    path = os.path.join(tmp, "body_visibility", f"{CLIP}.json")
    d = one_decision((5, 5))
    bv.atomic_write_json(path, d.as_dict())
    bv.atomic_write_json(path, d.as_dict())  # overwrite in place, no temp leftovers
    left = os.listdir(os.path.dirname(path))
    check("reporte json escrito atomico", json.load(open(path))["decision"] == "accepted"
          and left == [f"{CLIP}.json"], str(left))


# ----------------------------------------------------------- quarantine ----

def test_quarantine_moves_only_matching_clip():
    tmp = tempfile.mkdtemp(prefix="bv-q-")
    ds = os.path.join(tmp, "dataset")
    made = []
    for persons, clip in (("1_persons", CLIP), ("2_persons", CLIP), ("1_persons", "other_clip")):
        p = os.path.join(ds, persons, "parquet_files", clip + ".parquet")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write("x")
        made.append(p)
    moved = bv.quarantine_stale_parquet(ds, CLIP, "low_coverage")
    check("solo se mueve el parquet del clip", len(moved) == 2 and os.path.isfile(made[2])
          and not os.path.exists(made[0]) and not os.path.exists(made[1]), str(moved))
    check("destino fuera de *_persons", all("body_visibility" in m and "_persons/" not in m for m in moved),
          str(moved))
    moved2 = bv.quarantine_stale_parquet(ds, CLIP, "low_coverage")
    check("colision de nombres no sobrescribe", len(moved2) == 0 and len(set(moved)) == 2)
    # second pass with fresh copies lands next to the first, not on top
    for persons in ("1_persons",):
        p = os.path.join(ds, persons, "parquet_files", CLIP + ".parquet")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write("y")
    moved3 = bv.quarantine_stale_parquet(ds, CLIP, "low_coverage")
    check("re-cuarentena crea backup unico", len(moved3) == 1
          and len(os.listdir(os.path.join(ds, "body_visibility", "rejected", "low_coverage",
                                          "parquet_files"))) == 3, str(moved3))


def test_quarantine_is_a_noop_without_parquet():
    ds = os.path.join(tempfile.mkdtemp(prefix="bv-q0-"), "dataset")
    os.makedirs(os.path.join(ds, "1_persons", "parquet_files"), exist_ok=True)
    check("sin parquet no se crea nada", bv.quarantine_stale_parquet(ds, CLIP, "error") == []
          and not os.path.exists(os.path.join(ds, "body_visibility")))


# -------------------------------------------------------------- config ----

def test_config_validation():
    good = bv.VisibilityConfig.from_args("4,7", "40", "0.2")
    check("config por defecto 4,7 / 40 / 0.2",
          good.required_points == (4, 7) and good.min_visible_percent == 40.0
          and good.min_keypoint_confidence == 0.2)
    check("config default del modulo", bv.DEFAULT_REQUIRED_BODY_POINTS == (4, 7)
          and bv.DEFAULT_MIN_VISIBLE_PERCENT == 40.0
          and bv.DEFAULT_MIN_KEYPOINT_CONFIDENCE == 0.2)
    bad = {
        "lista vacia": ("", 40.0, 0.2),
        "indice fuera de BODY_25": ("25", 40.0, 0.2),
        "indice negativo": ("-1", 40.0, 0.2),
        "duplicado": ("4,4", 40.0, 0.2),
        "no numerico": ("hombro", 40.0, 0.2),
        "percent > 100": ("4", 100.5, 0.2),
        "percent < 0": ("4", -1, 0.2),
        "percent nan": ("4", float("nan"), 0.2),
        "percent inf": ("4", float("inf"), 0.2),
        "conf > 1": ("4", 40.0, 1.01),
        "conf negativa": ("4", 40.0, -0.1),
        "conf nan": ("4", 40.0, float("nan")),
    }
    for name, (p, pct, c) in bad.items():
        try:
            bv.VisibilityConfig.from_args(p, pct, c)
            check(f"config rechaza {name}", False, "aceptada")
        except bv.VisibilityConfigError:
            check(f"config rechaza {name}", True)
    try:
        bv.VisibilityConfig.from_args("0", 0.0, 0.0)
        check("extremos validos 0/0/0", True)
    except bv.VisibilityConfigError as e:
        check("extremos validos 0/0/0", False, str(e))


def test_expected_frames_must_be_positive():
    jd = build_json_dir(4, {4: 4, 7: 4})
    for bad in (0, -3, 2.5, float("nan"), True):
        try:
            bv.evaluate_body_visibility(jd, bad, cfg())
            check(f"expected_frames={bad} rechazado", False, "aceptado")
        except bv.VisibilityConfigError:
            check(f"expected_frames={bad} rechazado", True)


def test_archive_json_dir_preserves_previous_run():
    jd = build_json_dir(6, {4: 6, 7: 6}, overlap=True)
    with open(os.path.join(jd, "num_persons.txt"), "w") as fh:
        fh.write("1\n")
    archive = os.path.join(os.path.dirname(jd), "previous")
    moved = bv.archive_json_dir(jd, archive)
    check("archiva el JSON previo sin borrarlo",
          len(moved) == 6 and os.path.isdir(jd) and not os.listdir(jd)
          and len([n for n in os.listdir(archive) if n.endswith(bv.JSON_SUFFIX)]) == 6, str(moved[:2]))
    check("archiva tambien num_persons.txt", os.path.isfile(os.path.join(archive, "num_persons.txt")))
    check("archivar carpeta vacia no crea nada",
          bv.archive_json_dir(jd, os.path.join(os.path.dirname(jd), "otro")) == [])
    # a rerun that lands in the emptied folder is measured on its own frames only
    build_json_dir(6, {4: 6, 7: 6}, json_dir=jd, overlap=True)
    check("medicion sin mezclar la pasada anterior",
          bv.evaluate_body_visibility(jd, 6, cfg()).accepted)


def test_stale_json_blends_a_truncated_rerun():
    """Pins why the CLI archives previous JSON before an enabled rerun.

    Frame files from an older run are indistinguishable from fresh ones, so a
    truncated rerun that reuses the folder gets its frames blended with the old
    ones and can be accepted on measurements that are not from this run. The
    pure analyzer counts those frames honestly; the fix lives in the CLI, which
    moves the previous JSON aside (see test_argos_visibility.py).
    """
    jd = build_json_dir(10, {4: 10, 7: 10}, overlap=True)  # frames 0..9 of a good run
    for idx in range(4):  # an interrupted rerun rewrites only the first frames
        write_json(jd, idx, frame(idx, []))
    blended = bv.evaluate_body_visibility(jd, 10, cfg())
    check("JSON rancio reutilizado: la mezcla puede aceptar (riesgo documentado)",
          blended.accepted and blended.frames_no_person == 4, blended.reason)
    # with the folder archived, the same truncated rerun is measured on 4/10 frames
    fresh = make_json_dir()
    for idx in range(4):
        write_json(fresh, idx, frame(idx, []))
    honest = bv.evaluate_body_visibility(fresh, 10, cfg())
    check("sin rancio: el rerun truncado se mide sobre sus propios frames",
          not honest.accepted and honest.frames_missing_json == 6, honest.reason)


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[fn]()
        except Exception as e:  # noqa: BLE001
            check(f"{fn} (excepcion)", False, repr(e))
    print("\nTOTAL:", "FAIL " + ", ".join(FAIL) if FAIL else "todo OK")
    sys.exit(1 if FAIL else 0)
