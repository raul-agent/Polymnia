#!/usr/bin/env python3
"""Theia: ultima etapa del pipeline — anotacion visual por video con un VLM.

Para cada video que llego al final del pipeline (los `id` presentes en
`multi_data`), extrae 5 frames (primero, ultimo y 3 intermedios proporcionales)
y pregunta a un LLM con vision compatible con la API de OpenAI (vLLM) por 7
campos. Usa dos llamadas por video, cada una con las imagenes que le sirven:

  escena   -> frames del video RAW     (indoor_outdoor, show_type)
  hablante -> frames del MASKED + RAW  (hands_free, sitting_standing,
                                        screen_interaction, sex, age)

La salida se valida contra un esquema estricto; si un intento no lo cumple, se
reintenta devolviendole al modelo el motivo del fallo (`--attempts`, por
defecto 3). Ninguna anotacion invalida entra en las salidas.

Resultados:
  - CSV (por defecto dataset/{n}_persons/video_annotations.csv), una fila por id
    con columna `status` (ok|error). Sirve de punto de reanudacion: los ids con
    status=ok se saltan al relanzar.
  - DuckDB: columnas nuevas en `multi_data` (consultables junto al resto) y una
    tabla `video_annotations` con una fila por video.

Credenciales: --api-base/--api-key/--model o, si se omiten, las variables
THEIA_API_BASE / THEIA_API_KEY / THEIA_MODEL, leidas tambien de un .env en la
rai­z del repo (el .env esta excluido de git; usa .env.example como plantilla).

Pruebas: tools/test_theia.py (validador, muestreo de frames, limpieza de la
respuesta, y un e2e contra un servidor OpenAI-compatible falso).
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import duckdb
import requests

SCENE_FIELDS = ("indoor_outdoor", "show_type")
SPEAKER_FIELDS = ("hands_free", "sitting_standing", "screen_interaction", "sex", "age")
BOOL_FIELDS = ("hands_free", "screen_interaction")
FIELDS = SCENE_FIELDS + SPEAKER_FIELDS  # orden canonico de salidas

ENUMS = {
    "indoor_outdoor": ("indoor", "outdoor"),
    "show_type": ("news anchor", "monologue", "weather report", "interview"),
    "sitting_standing": ("sitting", "standing"),
    "sex": ("male", "female"),
}
NULLISH = {"", "null", "none", "n/a", "na", "unknown", "uncertain",
           "not sure", "cannot tell", "can't tell", "cant tell", "unclear",
           "i don't know", "i dont know", "i do not know", "no se", "no sé"}
TRUEISH = {"true", "yes", "si", "1"}
FALSEISH = {"false", "no", "0"}
AGE_RANGE = (0, 120)
# A reasoning model burns most of this on hidden chain-of-thought before the
# JSON: measured on a real vLLM endpoint, the speaker call (10 images) used 470
# reasoning tokens and needed ~540 in total. 400 truncated it to content=null.
DEFAULT_MAX_TOKENS = 2000
MAX_SIDE = 768  # frames reducidos a este lado maximo antes de mandar


class AnnotationError(ValueError):
    """La respuesta no cumple el esquema."""


# ------------------------------------------------------------------ env ----

def load_env(path: Path) -> None:
    """Carga KEY=VALUE de un .env sin pisar variables ya definidas."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


# ------------------------------------------------------------ esquema ----

def _clean(value):
    if isinstance(value, str):
        value = value.strip()
        low = value.lower()
        if low in NULLISH:
            return None
    return value


def validate_annotation(raw) -> dict:
    """Devuelve el dict canonico de 7 claves o lanza AnnotationError.

    Tolerante de forma (strings->bool/int, case/espacios, "uncertain"->None) y
    estricta de contenido: enums del dominio, edad entera en rango, claves que
    falten o sobren.
    """
    if not isinstance(raw, dict):
        raise AnnotationError(f"la respuesta no es un objeto JSON, es {type(raw).__name__}")
    missing = [k for k in FIELDS if k not in raw]
    extra = [k for k in raw if k not in FIELDS]
    if missing:
        raise AnnotationError(f"faltan claves: {missing}")
    if extra:
        raise AnnotationError(f"claves no permitidas: {extra}")

    out = {}
    for key in FIELDS:
        value = _clean(raw[key])
        if value is None:
            out[key] = None
            continue
        if key in BOOL_FIELDS:
            if isinstance(value, bool):
                out[key] = value
            elif isinstance(value, (int, float)) and value in (0, 1):
                out[key] = bool(value)
            elif isinstance(value, str) and value.lower() in TRUEISH | FALSEISH:
                out[key] = value.lower() in TRUEISH
            else:
                raise AnnotationError(f"{key}={raw[key]!r} no es booleano")
        elif key == "age":
            if isinstance(value, bool):
                raise AnnotationError("age no es un numero")
            if isinstance(value, str):
                if not re.fullmatch(r"\d{1,3}", value.strip()):
                    raise AnnotationError(f"age={value!r} no es una edad entera")
                value = int(value)
            elif isinstance(value, float):
                if not value.is_integer():
                    raise AnnotationError(f"age={value!r} no es entera")
                value = int(value)
            if not isinstance(value, int) or not AGE_RANGE[0] <= value <= AGE_RANGE[1]:
                raise AnnotationError(f"age={value!r} fuera de {AGE_RANGE}")
            out[key] = value
        else:
            allowed = ENUMS[key]
            if not isinstance(value, str):
                raise AnnotationError(f"{key}={value!r} no es texto")
            canon = {e.lower(): e for e in allowed}
            low = value.lower().strip()
            if low not in canon:
                raise AnnotationError(
                    f"{key}={value!r} no esta en {allowed} (usa null si no se puede saber)")
            out[key] = canon[low]
    return out


# -------------------------------------------------------------- frames ----

def pick_frame_indices(total: int) -> list[int]:
    """5 indices: primero, ultimo y 3 intermedios proporcionales, sin duplicados."""
    if total <= 0:
        return []
    idx = sorted({min(total - 1, int(i * (total - 1) / 4)) for i in range(5)})
    return idx


def extract_frames(video_path: Path, indices: list[int]) -> list[bytes]:
    """Decodifica solo los frames pedidos y los devuelve como PNG (JPEG-serializable)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        out = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = MAX_SIDE / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                out.append(buf.tobytes())
        return out
    finally:
        cap.release()


def data_url(image_bytes: list[bytes]) -> list[dict]:
    return [{"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64,"
                          + base64.b64encode(b).decode()}} for b in image_bytes]


# -------------------------------------------------------- prompts/LLM ----

JSON_RULES = (
    "Responde UNICAMENTE con un objeto JSON, sin texto alrededor ni bloque de codigo. "
    "Incluye EXACTAMENTE estas 7 claves: "
    "indoor_outdoor: \"indoor\" u \"outdoor\"; "
    "show_type: una de \"news anchor\", \"monologue\", \"weather report\", \"interview\"; "
    "hands_free: true|false (booleano, sin comillas); "
    "sitting_standing: \"sitting\" o \"standing\"; "
    "screen_interaction: true|false (booleano, sin comillas); "
    "sex: \"male\" o \"female\"; age: edad aproximada como numero entero anios. "
    "Para cualquier clave que NO puedas juzgar con estas imagenes, usa null. No inventes."
)

SCENE_PROMPT = (
    "Estas son 5 capturas de un clip de television, en orden cronologico "
    "(primero, tres intermedios proporcionales y ultimo). " + JSON_RULES +
    " De ti solo importan, juzgadas sobre el estudio/escena que se ve: "
    "indoor_outdoor y show_type (el formato del programa: presentador de telediario, "
    "monologo, parte meteorologico o entrevista). El resto de claves ponlas a null."
)

SPEAKER_PROMPT = (
    "Las primeras 5 imagenes son fotogramas de un clip con TODO menos al hablante "
    "enmascarado (solo se ve a la persona que habla); las 5 ultimas son las mismas "
    "posiciones del clip completo, para contexto. En ambos grupos el orden es cronologico. "
    + JSON_RULES +
    " De ti solo importan, referidas a ESE hablante: hands_free (si se le ven las manos "
    "libres, sin sostener nada), sitting_standing, screen_interaction (si toca, senala o "
    "opera una pantalla visible), sex y age aproximada. El resto de claves ponlas a null."
)


def ask_vlm(session: requests.Session, api_base: str, api_key: str, model: str,
            images: list[dict], prompt: str, timeout: int = 600,
            max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    resp = session.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": images + [
                {"type": "text", "text": prompt}]}],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    content = choice["message"].get("content")
    if not content:
        # A reasoning model that spends the whole budget on `reasoning_content`
        # returns content=null with finish_reason=length. Saying "no JSON in the
        # response" sends you hunting for a formatting bug; the real cause is the
        # token budget, so name it.
        reason = choice.get("finish_reason")
        if reason == "length":
            raise AnnotationError(
                f"respuesta truncada (finish_reason=length) con max_tokens={max_tokens}: "
                "el modelo agoto el presupuesto antes de escribir el JSON; sube --max-tokens")
        raise AnnotationError(f"respuesta vacia (finish_reason={reason!r})")
    return content


def parse_json_response(text: str) -> dict:
    """Acepta JSON puro, cercas ``` o texto alrededor; exige objeto."""
    if not isinstance(text, str):
        raise AnnotationError("respuesta sin texto")
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            raise AnnotationError("no hay objeto JSON en la respuesta")
        try:
            data = json.loads(body[start:end + 1])
        except json.JSONDecodeError as e:
            raise AnnotationError(f"JSON mal formado: {e}")
    if not isinstance(data, dict):
        raise AnnotationError("la respuesta no es un objeto JSON")
    return data


# -------------------------------------------------------- por video ----

def annotate_video(video_id: str, raw_video: Path, masked_video: Path | None,
                   api_base: str, api_key: str, model: str, attempts: int,
                   session: requests.Session | None = None,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """Devuelve dict de 7 claves combinando escena (raw) y hablante (masked+raw).

    Lanza AnnotationError si una de las dos partes agota los intentos.
    """
    session = session or requests.Session()
    probe = cv2.VideoCapture(str(raw_video))
    try:
        total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        probe.release()
    indices = pick_frame_indices(total)
    if not indices:
        raise AnnotationError(f"{video_id}: no se pudo leer {raw_video}")

    parts = [(SCENE_PROMPT, [raw_video]),
             (SPEAKER_PROMPT, [v for v in (masked_video, raw_video) if v and v.is_file()])]
    merged: dict = {k: None for k in FIELDS}
    for prompt, sources in parts:
        images = []
        for src in sources:
            images += data_url(extract_frames(src, indices))
        last_err = "sin intentos"
        feedback = ""
        budget = max_tokens
        for _ in range(attempts):
            try:
                text = ask_vlm(session, api_base, api_key, model, images,
                               prompt + feedback, max_tokens=budget)
                part = validate_annotation(parse_json_response(text))
                part = validate_annotation(parse_json_response(text))
                # cada parte trae nulls en los campos de la otra: solo se rellenan
                # valores, nunca se pisan con nulls ajenos
                for k, v in part.items():
                    if v is not None:
                        merged[k] = v
                break
            except (AnnotationError, requests.RequestException) as e:
                last_err = str(e)
                if "finish_reason=length" in last_err:
                    # same prompt with the same budget would truncate the same way
                    budget *= 2
                feedback = (f"\n\nTu respuesta anterior fue INVALIDA ({last_err[:300]}). "
                            "Corrigela y responde otra vez solo con el JSON.")
        else:
            raise AnnotationError(f"{video_id}: intentos agotados ({last_err})")
    return merged


# ------------------------------------------------------------- dataset ----

def _ids_in_db(db_path: Path) -> list[str]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT id FROM multi_data ORDER BY id").fetchall()]
    finally:
        con.close()


def _read_csv(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", *FIELDS, "status", "error"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)


def _fmt(value):
    return "" if value is None else str(value)


def sync_to_db(db_path: Path, rows: list[dict]) -> None:
    """Vuelca las anotaciones a multi_data (columnas) y a video_annotations."""
    con = duckdb.connect(str(db_path))
    try:
        types = {k: ("BOOLEAN" if k in BOOL_FIELDS else
                     "INTEGER" if k == "age" else "VARCHAR") for k in FIELDS}
        for key, sql_type in types.items():
            con.execute(f"ALTER TABLE multi_data ADD COLUMN IF NOT EXISTS {key} {sql_type}")
        con.execute("""CREATE TABLE IF NOT EXISTS video_annotations (
            id VARCHAR PRIMARY KEY, {}, status VARCHAR, error VARCHAR)"""
            .format(", ".join(f"{k} {types[k]}" for k in FIELDS)))

        for row in rows:
            values = []
            for k in FIELDS:
                raw = row.get(k, "")
                if raw == "":
                    values.append(None)
                elif k in BOOL_FIELDS:
                    values.append(raw.lower() in TRUEISH)
                elif k == "age":
                    values.append(int(raw))
                else:
                    values.append(raw)
            con.execute("UPDATE multi_data SET " + ", ".join(f"{k} = ?" for k in FIELDS)
                        + " WHERE id = ?", values + [row["id"]])
            con.execute("INSERT OR REPLACE INTO video_annotations VALUES (" +
                        ",".join(["?"] * (3 + len(FIELDS))) + ")",  # id + 7 + status + error
                        [row["id"], *values, row.get("status", ""), row.get("error", "")])
    finally:
        con.close()


def annotate_dataset(input_folder, n_persons, db_name, api_base, api_key, model,
                     attempts=3, workers=4, csv_path=None, limit=None, write_db=True,
                     max_tokens=DEFAULT_MAX_TOKENS):
    """Anota todos los ids de multi_data. Devuelve (ok, error).

    write_db=False deja la base intacta: solo se lee para saber que ids anotar y
    el resultado queda en el CSV.
    """
    input_folder = Path(input_folder)
    n_dir = f"{n_persons}_persons"
    db_path = Path(db_name)
    if not db_path.is_file():
        raise FileNotFoundError(f"no existe la base {db_path}; ejecuta antes `polymnia merge`")
    csv_path = Path(csv_path) if csv_path else \
        input_folder / "dataset" / n_dir / "video_annotations.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(csv_path)
    done = {r["id"] for r in rows if r.get("status") == "ok"}
    rows = [r for r in rows if r.get("status") == "ok"]  # los error se reintentan

    todo = [i for i in _ids_in_db(db_path) if i not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        print("[theia] nada que anotar (todo esta ya en el CSV)")
        return len(rows), 0

    raw_dir = input_folder / "videos" / "raw"
    masked_dir = input_folder / "videos" / "masked"
    errors: dict[str, str] = {}
    lock_rows = list(rows)

    def work(video_id: str):
        raw_video = raw_dir / f"{video_id}.mp4"
        if not raw_video.is_file():
            raise AnnotationError(f"no existe {raw_video}")
        return annotate_video(video_id, raw_video, masked_dir / f"{video_id}.mp4",
                              api_base, api_key, model, attempts, max_tokens=max_tokens)

    print(f"[theia] anotando {len(todo)} videos ({len(done)} ya hechos), "
          f"model={model} @ {api_base}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, vid): vid for vid in todo}
        for future in futures:
            vid = futures[future]
            try:
                ann = future.result()
                lock_rows.append({"id": vid, **{k: _fmt(ann[k]) for k in FIELDS},
                                  "status": "ok", "error": ""})
            except (AnnotationError, FileNotFoundError) as e:
                errors[vid] = str(e)
                lock_rows.append({"id": vid, **{k: "" for k in FIELDS},
                                  "status": "error", "error": str(e)})
            _write_csv(csv_path, lock_rows)  # avance persistente: resume tras kill
            n_ok = sum(1 for r in lock_rows if r["status"] == "ok")
            print(f"  [{n_ok}/{len(todo) + len(done)}] {vid}: "
                  f"{'ok' if errors.get(vid) is None else 'ERROR ' + errors[vid][:120]}")

    lock_rows.sort(key=lambda r: r["id"])
    _write_csv(csv_path, lock_rows)
    if write_db:
        sync_to_db(db_path, lock_rows)
    n_err = sum(1 for r in lock_rows if r["status"] == "error")
    print(f"[theia] ok={sum(1 for r in lock_rows if r['status'] == 'ok')} error={n_err} "
          f"csv={csv_path}")
    return len([r for r in lock_rows if r["status"] == "ok"]), n_err


# ---------------------------------------------------------------- cli ----

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="sin --api-base/--api-key/--model se usan THEIA_API_BASE/THEIA_API_KEY/"
               "THEIA_MODEL del entorno o del .env de la raiz del repo")
    parser.add_argument("-i", "--input_folder", required=True,
                        help="carpeta del pipeline (con dataset/ y videos/)")
    parser.add_argument("-n", "--n_persons", type=int, required=True)
    parser.add_argument("-d", "--db_name", default=None,
                        help="DuckDB ya cargado (el mismo de `merge`)")
    parser.add_argument("--csv", default=None, help="ruta del CSV de anotaciones")
    parser.add_argument("--api-base", default=os.environ.get("THEIA_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("THEIA_API_KEY", "none"))
    parser.add_argument("--model", default=os.environ.get("THEIA_MODEL"))
    parser.add_argument("--attempts", type=int, default=3,
                        help="reintentos de respuesta invalida por parte (defecto 3)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"presupuesto por llamada (defecto {DEFAULT_MAX_TOKENS}); "
                             "los modelos de razonamiento lo gastan en su cadena "
                             "oculta y se quedan sin espacio para el JSON")
    parser.add_argument("--no-db", action="store_true", help="solo CSV, no tocar DuckDB")
    args = parser.parse_args()

    load_env(Path(__file__).resolve().parent / ".env")
    args.api_base = args.api_base or os.environ.get("THEIA_API_BASE")
    args.api_key = args.api_key or os.environ.get("THEIA_API_KEY", "none")
    args.model = args.model or os.environ.get("THEIA_MODEL")
    if not args.api_base or not args.model:
        parser.error("falta el endpoint: define THEIA_API_BASE y THEIA_MODEL "
                     "(.env o --api-base/--model)")

    db_name = args.db_name or str(Path(args.input_folder) / "dataset"
                                  / f"{args.n_persons}_persons" / "dataset.duckdb")
    _, n_err = annotate_dataset(
        args.input_folder, args.n_persons, db_name, args.api_base, args.api_key,
        args.model, attempts=args.attempts, workers=args.workers,
        csv_path=args.csv, limit=args.limit, write_db=not args.no_db,
        max_tokens=args.max_tokens)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
