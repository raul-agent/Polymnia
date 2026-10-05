"""Tests for theia.py, the vision-annotation stage.

The stage must never let an invalid annotation through: the acceptance
criterion is syntactically valid JSON that fits the strict schema, enforced by
the retry loop. These tests pin the schema validator, the frame sampling, the
LLM-response cleaning, and one end-to-end run against a fake OpenAI-compatible
server (the first response is deliberately broken to prove the retry works).
"""
import base64
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import theia  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- schema ----

GOOD = {
    "indoor_outdoor": "indoor",
    "show_type": "news anchor",
    "hands_free": True,
    "sitting_standing": "sitting",
    "screen_interaction": False,
    "sex": "female",
    "age": 41,
}


def test_validate_ok():
    out = theia.validate_annotation(dict(GOOD))
    check("valido pasa intacto", out == GOOD, str(out))


def test_validate_coercions():
    raw = dict(GOOD, hands_free="true", screen_interaction="False", age="41",
               indoor_outdoor=" INDOOR ")
    out = theia.validate_annotation(raw)
    ok = (out["hands_free"] is True and out["screen_interaction"] is False
          and out["age"] == 41 and out["indoor_outdoor"] == "indoor")
    check("coercion (str bool, str int, case/espacios)", ok, str(out))


def test_validate_uncertain_becomes_null():
    raw = dict(GOOD, hands_free="uncertain", sex="I don't know",
               show_type="uncertain", age=None)
    out = theia.validate_annotation(raw)
    ok = (out["hands_free"] is None and out["sex"] is None
          and out["show_type"] is None and out["age"] is None)
    check("uncertain/desconocido -> None", ok, str(out))


def test_validate_rejects_garbage():
    cases = {
        "enum invalido": dict(GOOD, indoor_outdoor="space"),
        "falta clave": {k: v for k, v in GOOD.items() if k != "age"},
        "clave extra": dict(GOOD, humor="dry"),
        "edad fuera de rango": dict(GOOD, age=150),
        "edad no entera": dict(GOOD, age="middle-aged"),
        "bool no booleanable": dict(GOOD, hands_free="maybe"),
        "no es dict": ["indoor"],
    }
    n_ok = 0
    for name, bad in cases.items():
        try:
            theia.validate_annotation(bad)
            rejected = False
        except theia.AnnotationError:
            rejected = True
        n_ok += rejected
        if not rejected:
            print(f"    no rechazo: {name}")
    check(f"rechaza {len(cases)} variantes invalidas", n_ok == len(cases), f"{n_ok}/{len(cases)}")


# ---------------------------------------------------------------- frames ----

def test_pick_frame_indices():
    check("clip largo proporcional", theia.pick_frame_indices(600) == [0, 149, 299, 449, 599],
          str(theia.pick_frame_indices(600)))
    check("clip exacto de 5", theia.pick_frame_indices(5) == [0, 1, 2, 3, 4],
          str(theia.pick_frame_indices(5)))
    idx = theia.pick_frame_indices(3)
    check("clip corto sin duplicados", idx == [0, 1, 2] and len(idx) == len(set(idx)), str(idx))
    check("un frame", theia.pick_frame_indices(1) == [0], str(theia.pick_frame_indices(1)))


# ------------------------------------------------------- response parse ----

def test_parse_json_response():
    fenced = "```json\n" + json.dumps(GOOD) + "\n```"
    check("fence markdown", theia.parse_json_response(fenced) == GOOD)
    wrapped = "Here is the annotation you asked for:\n" + json.dumps(GOOD) + "\nHope this helps!"
    check("texto alrededor", theia.parse_json_response(wrapped) == GOOD)
    for bad in ("", "no hay json aqui", "{incompleto", json.dumps([1, 2])):
        try:
            theia.parse_json_response(bad)
            check(f"rechaza '{bad[:18]}'", False)
        except theia.AnnotationError:
            check(f"rechaza '{bad[:18]}'", True)


# ---------------------------------------------------------------- e2e ------

class FakeVLM:
    """OpenAI-compatible /v1/chat/completions with scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                outer.requests.append((self.path, body))
                reply = outer.replies.pop(0) if outer.replies else "{}"
                if isinstance(reply, dict) and "_raw" in reply:
                    payload = json.dumps(reply["_raw"]).encode()
                else:
                    payload = json.dumps(
                        {"choices": [{"message": {"content": reply if isinstance(reply, str)
                                                  else json.dumps(reply)}}]}).encode()
                self.send_response(200)
                self.headers_sent = True
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self):
        self.server.shutdown()


def make_video(path, frames=12, color=(40, 90, 160)):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    for i in range(frames):
        img = np.zeros((64, 96, 3), dtype=np.uint8)
        img[:, :] = color
        img[i * 3:i * 3 + 4, 10:40] = (255, 255, 255)  # marca movediza
        vw.write(img)
    vw.release()


def test_end_to_end_with_fake_server():
    import csv as csvmod
    import duckdb

    scene = {"indoor_outdoor": "indoor", "show_type": "news anchor",
             "hands_free": None, "sitting_standing": None,
             "screen_interaction": None, "sex": None, "age": None}
    speaker = {"indoor_outdoor": None, "show_type": None,
               "hands_free": True, "sitting_standing": "sitting",
               "screen_interaction": False, "sex": "female", "age": 41}

    with tempfile.TemporaryDirectory() as td:
        root = theia.Path(td)
        raw = root / "videos" / "raw"
        masked = root / "videos" / "masked"
        raw.mkdir(parents=True)
        masked.mkdir(parents=True)
        make_video(raw / "clipA.mp4")
        make_video(masked / "clipA.mp4", color=(200, 30, 30))
        db_path = root / "d.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute("CREATE TABLE multi_data (id VARCHAR, frame BIGINT)")
        con.execute("INSERT INTO multi_data VALUES ('clipA', 0), ('clipA', 1)")
        con.close()

        # el servidor responde: 2 JSON invalidos, luego los validos (escena/hablante)
        fake = FakeVLM(["no es json", "{incompleto", scene, speaker])
        try:
            csv_path = root / "ann.csv"
            n_ok, n_err = theia.annotate_dataset(
                input_folder=root, n_persons=1, db_name=db_path,
                api_base=fake.base_url, api_key="test", model="fake-vlm",
                attempts=3, workers=1, csv_path=csv_path,
            )
            rows = list(csvmod.DictReader(open(csv_path)))
            check("e2e: 1 ok 0 error", (n_ok, n_err) == (1, 0), str((n_ok, n_err)))
            check("e2e: fila csv con valores combinados",
                  len(rows) == 1 and rows[0]["id"] == "clipA"
                  and rows[0]["indoor_outdoor"] == "indoor"
                  and rows[0]["show_type"] == "news anchor"
                  and rows[0]["hands_free"] == "True"
                  and rows[0]["sex"] == "female" and rows[0]["age"] == "41",
                  json.dumps(rows))
            check("e2e: reintentos ocurrieron (4 peticiones: 2 malas + 2 etapas)",
                  len(fake.requests) == 4, str(len(fake.requests)))
            paths = {p for p, _ in fake.requests}
            check("e2e: endpoint /v1/chat/completions",
                  all(p.endswith("/v1/chat/completions") for p in paths), str(paths))
            body = fake.requests[-1][1]
            kinds = [c["type"] for c in body["messages"][-1]["content"]]
            n_img = kinds.count("image_url")
            check("e2e: 10 imagenes en la peticion de hablante (5 masked + 5 raw)",
                  n_img == 10, f"{n_img} imgs, kinds={kinds[:4]}...")
            body_scene = fake.requests[-2][1]
            kinds_scene = [c["type"] for c in body_scene["messages"][-1]["content"]]
            check("e2e: la escena solo ve los 5 frames del raw",
                  kinds_scene.count("image_url") == 5, str(kinds_scene.count("image_url")))
            b64 = [c["image_url"]["url"] for c in body["messages"][-1]["content"]
                   if c["type"] == "image_url"][0]
            ok_b64 = b64.startswith("data:image/") and \
                len(base64.b64decode(b64.split(",", 1)[1])) > 100
            check("e2e: frames como data-url de imagen", ok_b64)

            con = duckdb.connect(str(db_path))
            cols = [r[0] for r in con.execute("DESCRIBE multi_data").fetchall()]
            check("e2e: columnas anadidas a multi_data",
                  {"indoor_outdoor", "show_type", "hands_free", "sitting_standing",
                   "screen_interaction", "sex", "age"} <= set(cols), str(cols))
            row = con.execute("""SELECT indoor_outdoor, show_type, hands_free, sex, age
                                 FROM multi_data WHERE id='clipA' LIMIT 1""").fetchone()
            check("e2e: valores consultables en multi_data",
                  row == ("indoor", "news anchor", True, "female", 41), str(row))
            nv = con.execute("SELECT count(*) FROM video_annotations").fetchone()[0]
            check("e2e: tabla video_annotations", nv == 1, str(nv))
            con.close()

            # re-ejecucion: debe saltar el id ya anotado (resume) sin nuevas llamadas
            reqs_before = len(fake.requests)
            n2_ok, _ = theia.annotate_dataset(
                input_folder=root, n_persons=1, db_name=db_path,
                api_base=fake.base_url, api_key="test", model="fake-vlm",
                attempts=1, workers=1, csv_path=csv_path)
            check("e2e: resume no repite ids (cero llamadas nuevas)",
                  n2_ok == 1 and len(fake.requests) == reqs_before,
                  f"ok={n2_ok} reqs={len(fake.requests) - reqs_before} nuevas")
        finally:
            fake.stop()


def test_no_db_leaves_database_untouched():
    """--no-db debe dar CSV sin tocar el DuckDB (ni columnas ni filas)."""
    import duckdb

    scene = {"indoor_outdoor": "outdoor", "show_type": "monologue", "hands_free": None,
             "sitting_standing": None, "screen_interaction": None, "sex": None, "age": None}
    speaker = {"indoor_outdoor": None, "show_type": None, "hands_free": False,
               "sitting_standing": "standing", "screen_interaction": True,
               "sex": "male", "age": 30}
    with tempfile.TemporaryDirectory() as td:
        root = theia.Path(td)
        (root / "videos" / "raw").mkdir(parents=True)
        make_video(root / "videos" / "raw" / "clipC.mp4")
        db = root / "c.duckdb"
        con = duckdb.connect(str(db))
        con.execute("CREATE TABLE multi_data (id VARCHAR, frame BIGINT)")
        con.execute("INSERT INTO multi_data VALUES ('clipC', 0)")
        cols_before = [r[0] for r in con.execute("DESCRIBE multi_data").fetchall()]
        tables_before = sorted(r[0] for r in con.execute("SHOW TABLES").fetchall())
        con.close()

        fake = FakeVLM([scene, speaker])
        try:
            n_ok, _ = theia.annotate_dataset(
                input_folder=root, n_persons=1, db_name=db, api_base=fake.base_url,
                api_key="t", model="m", attempts=2, workers=1,
                csv_path=root / "c.csv", write_db=False)
        finally:
            fake.stop()
        con = duckdb.connect(str(db))
        cols_after = [r[0] for r in con.execute("DESCRIBE multi_data").fetchall()]
        tables_after = sorted(r[0] for r in con.execute("SHOW TABLES").fetchall())
        con.close()
        check("no-db: anota igualmente (1 ok)", n_ok == 1, str(n_ok))
        check("no-db: CSV escrito", (root / "c.csv").is_file())
        check("no-db: multi_data sin columnas nuevas", cols_after == cols_before,
              f"antes={cols_before} despues={cols_after}")
        check("no-db: ni una tabla nueva", tables_after == tables_before,
              f"antes={tables_before} despues={tables_after}")


def test_exhausted_retries_record_error():
    import csv as csvmod

    with tempfile.TemporaryDirectory() as td:
        root = theia.Path(td)
        raw = root / "videos" / "raw"
        raw.mkdir(parents=True)
        make_video(raw / "clipB.mp4")
        # sin db: --no-db implicito cuando db_name no existe aun? usamos db de verdad
        db = root / "e.duckdb"
        import duckdb
        con = duckdb.connect(str(db))
        con.execute("CREATE TABLE multi_data (id VARCHAR, frame BIGINT)")
        con.execute("INSERT INTO multi_data VALUES ('clipB', 0)")
        con.close()
        fake = FakeVLM(["basura"] * 3 + ["mas basura"] * 3)
        try:
            csv_path = root / "e.csv"
            n_ok, n_err = theia.annotate_dataset(
                input_folder=root, n_persons=1, db_name=db, api_base=fake.base_url,
                api_key="t", model="m", attempts=3, workers=1, csv_path=csv_path)
            rows = list(csvmod.DictReader(open(csv_path)))
            check("agotado: 0 ok 1 error", (n_ok, n_err) == (0, 1), str((n_ok, n_err)))
            check("agotado: fila con status=error y valores vacios",
                  len(rows) == 1 and rows[0]["status"] == "error"
                  and rows[0]["indoor_outdoor"] == "", json.dumps(rows))
        finally:
            fake.stop()


def test_reasoning_model_truncated_is_reported():
    """Un modelo de razonamiento puede agotar max_tokens en reasoning y dejar
    content vacio: el error debe decirlo (finish_reason), no 'no hay JSON'."""
    import csv as csvmod
    length_stop = {"_raw": {"choices": [{"message": {"role": "assistant", "content": None},
                                         "finish_reason": "length"}]}}
    with tempfile.TemporaryDirectory() as td:
        root = theia.Path(td)
        (root / "videos" / "raw").mkdir(parents=True)
        make_video(root / "videos" / "raw" / "clipD.mp4")
        db = root / "d4.duckdb"
        import duckdb
        con = duckdb.connect(str(db))
        con.execute("CREATE TABLE multi_data (id VARCHAR, frame BIGINT)")
        con.execute("INSERT INTO multi_data VALUES ('clipD', 0)")
        con.close()
        fake = FakeVLM([length_stop] * 6)
        try:
            csv_path = root / "d4.csv"
            n_ok, n_err = theia.annotate_dataset(
                input_folder=root, n_persons=1, db_name=db, api_base=fake.base_url,
                api_key="t", model="m", attempts=3, workers=1, csv_path=csv_path)
            row = next(iter(csvmod.DictReader(open(csv_path))))
            check("truncado: termina en error", (n_ok, n_err) == (0, 1), str((n_ok, n_err)))
            check("truncado: el error menciona finish_reason/tokens",
                  "finish_reason=length" in row["error"], row["error"][:160])
            # y con presupuesto mayor debe pasar (el servidor responde bien tras el truncamiento)
            scene = {"indoor_outdoor": "outdoor", "show_type": "interview",
                     "hands_free": None, "sitting_standing": None,
                     "screen_interaction": None, "sex": None, "age": None}
            speaker = {"indoor_outdoor": None, "show_type": None, "hands_free": False,
                       "sitting_standing": "standing", "screen_interaction": False,
                       "sex": "male", "age": 50}
            fake2 = FakeVLM([length_stop, scene, speaker])
            try:
                n2, _ = theia.annotate_dataset(
                    input_folder=root, n_persons=1, db_name=db, api_base=fake2.base_url,
                    api_key="t", model="m", attempts=3, workers=1,
                    csv_path=root / "d5.csv", max_tokens=2000)
                check("truncado: el retry dentro del intento recupera", n2 == 1, str(n2))
                check("truncado: max_tokens viaja en la peticion",
                      fake2.requests[0][1].get("max_tokens") == 2000,
                      str(fake2.requests[0][1].get("max_tokens")))
            finally:
                fake2.stop()
        finally:
            fake.stop()


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[fn]()
        except Exception as e:
            check(f"{fn} (excepcion)", False, repr(e))
    print("\nTOTAL:", "FAIL " + ", ".join(FAIL) if FAIL else "todo OK")
    sys.exit(1 if FAIL else 0)
