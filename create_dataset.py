#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de creación de datasets multimodales:

- Descarga clips y procesa con TalkNet‑ASD.
- Filtra a un solo hablante y ejecuta Argos (OpenPose + dfMaker).
- Genera parquet, análisis acústico (ecos / apate) y DuckDB.
- Admite reanudación (--resume) y arranque directo en Argos (--start_at_argos).

Autor: DaedalusLAB
"""

import argparse
import os
import concurrent.futures
import sys
import re 

# Ruta del propio script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHON_SCRIPT_DIR"] = script_dir  # para los scripts R

# Ajusta este path a tu instalación de TalkNet‑ASD
ASD_PATH = "/home/brian/TalkNet-ASD"

# ---------- utilidades comunes ------------------------------------------------

def run_command(command: str) -> int:
    """Ejecuta un comando shell y devuelve su código de salida."""
    print(f"Ejecutando: {command}")
    return os.system(command)

def check_json_generated(output_folder: str) -> bool:
    """Devuelve True si existen carpetas JSON_FILES generadas por OpenPose."""
    openpose_dir = os.path.join(output_folder, "dataset", "OpenPose")
    if not os.path.exists(openpose_dir):
        return False
    return any(
        os.path.isdir(os.path.join(openpose_dir, vid, "JSON_FILES"))
        for vid in os.listdir(openpose_dir)
    )

def check_parquet_generated(output_folder: str) -> bool:
    """Devuelve True si ya hay archivos .parquet generados."""
    parquet_dir = os.path.join(output_folder, "dataset", "1_persons", "parquet_files")
    return os.path.exists(parquet_dir) and any(
        f.endswith(".parquet") for f in os.listdir(parquet_dir)
    )

def mark_apate_ecos_done(output_folder: str) -> None:
    """Crea los ficheros bandera que indican que ecos.py y apate.py terminaron bien."""
    for flag in ("apate_done.txt", "ecos_done.txt"):
        with open(os.path.join(output_folder, flag), "w") as f:
            f.write("done")

def sanitize_filename(name: str) -> str:
    """Reemplaza caracteres problemáticos para nombres de archivo."""
    return re.sub(r"[^\w\-]", "_", name)

# ---------- pasos condicionales ----------------------------------------------

def run_argos_if_needed(output_folder: str) -> None:
    """
    Ejecuta argos.py (OpenPose + dfMaker) si no existen JSON_FILES.
    Es idempotente: si los JSON ya están, se omite.
    """
    if check_json_generated(output_folder):
        print("Las carpetas JSON_FILES existen. Argos no es necesario.")
        return

    print("No se encontraron JSON_FILES. Lanzando argos.py...")
    run_command(
        "python3 argos.py "
        f'--videos_folder "{os.path.join(output_folder, "videos", "1_person")}" '
        f'--output_folder "{output_folder}" '
        "--openpose_path /opt/openpose "
        "--face_hands True "
        "--skeletons True"
    )

def generate_parquet_if_needed(output_folder: str) -> None:
    """
    Genera los .parquet mediante max_people_classification.R
    si hay JSON pero aún no existen los parquet.
    """
    if not check_json_generated(output_folder):
        print("Aún no hay JSON; no se puede generar parquet.")
        return
    if check_parquet_generated(output_folder):
        print("Parquet ya generados. Se omite Rscript.")
        return

    print("Generando archivos parquet con max_people_classification.R...")
    dataset_dir = os.path.join(output_folder, "dataset")
    openpose_dir = os.path.join(dataset_dir, "OpenPose")
    for video_name in os.listdir(openpose_dir):
        json_folder = os.path.join(openpose_dir, video_name, "JSON_FILES")
        if os.path.exists(json_folder):
            print(f"→ Rscript en {json_folder}")
            run_command(
                f'Rscript max_people_classification.R "{json_folder}" "{dataset_dir}"'
            )

def run_missing_voice_scripts(output_folder: str) -> None:
    """
    Ejecuta ecos.py y/o apate.py si aún no tienen su bandera 'done'.
    """
    tasks = []
    for name in ("apate", "ecos"):
        flag = os.path.join(output_folder, f"{name}_done.txt")
        if not os.path.exists(flag):
            cmd = f'python3 {name}.py --input_folder {output_folder} --n_persons 1'
            tasks.append((name, cmd, flag))

    if not tasks:
        print("ecos.py y apate.py ya están marcados como completados.")
        return

    print("Ejecutando análisis de voz faltantes...")
    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = {ex.submit(run_command, cmd): (name, flag) for name, cmd, flag in tasks}
        for fut in concurrent.futures.as_completed(futures):
            name, flag = futures[fut]
            if fut.result() == 0:
                print(f"{name}.py terminó correctamente.")
                with open(flag, "w") as f:
                    f.write("done")
            else:
                print(f"⚠️  {name}.py falló (no se crea flag).")

def build_duckdb_if_needed(output_folder: str) -> None:
    """
    Genera dataset.duckdb si aún no existe, previa confirmación interactiva.
    """
    db_path = os.path.join(output_folder, "dataset.duckdb")
    if os.path.exists(db_path):
        print("dataset.duckdb ya existe.")
        return
    ans = input("¿Deseas generar dataset.duckdb ahora? (s/n): ").strip().lower()
    if ans == "s":
        run_command(
            f'python3 hefesto.py --input_folder "{output_folder}" --n_persons 1 '
            f'--db_name "{db_path}"'
        )

# ---------- rutinas de recuperación ------------------------------------------

def run_recovery(output_folder: str) -> None:
    """Ruta de recuperación (`--resume`)."""
    run_argos_if_needed(output_folder)
    generate_parquet_if_needed(output_folder)
    if check_parquet_generated(output_folder):
        run_missing_voice_scripts(output_folder)
    else:
        print("Sin parquet: ecos/apate no pueden ejecutarse todavía.")
    build_duckdb_if_needed(output_folder)

# ---------- flujo principal ---------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Pipeline de creación de datasets multimodales."
    )
    p.add_argument("--input", required=True, help="CSV generado por tabulate")
    p.add_argument("--output_folder", required=True, help="Carpeta de salida")
    p.add_argument("--searchterm", required=True, help="Término de búsqueda")
    p.add_argument("--offset", type=int, default=2, help="Offset en segundos")
    p.add_argument("--check_hands", action="store_true", help="Comprobar manos")
    p.add_argument("--resume", action="store_true", help="Reanudar proceso")
    p.add_argument(
        "--start_at_argos",
        action="store_true",
        help="Arrancar directamente en el paso Argos/OpenPose",
    )
    p.add_argument(
        "--skip_download",
        action="store_true",
        help="Suponer que los vídeos ya están en videos/raw",
    )
    args = p.parse_args()

    out_dir = args.output_folder
    os.makedirs(out_dir, exist_ok=True)

    # ---------- modo recuperación --------------------------------------------
    if args.resume:
        print("==> Modo recuperación (--resume)")
        run_recovery(out_dir)
        return

    # ---------- modo arranque en Argos ---------------------------------------
    if args.start_at_argos:
        print("==> Arranque directo en Argos (--start_at_argos)")
        run_argos_if_needed(out_dir)
        generate_parquet_if_needed(out_dir)
        run_missing_voice_scripts(out_dir)

        # Construcción de DuckDB con nombre basado en searchterm
        db_filename = f"{sanitize_filename(args.searchterm)}.duckdb"
        duckdb_path = os.path.join(out_dir, db_filename)
        if not os.path.exists(duckdb_path):
            print(f"Generando {db_filename} directamente (modo --start_at_argos)…")
            run_command(
                f'python3 hefesto.py --input_folder "{out_dir}" '
                f'--n_persons 1 --db_name "{duckdb_path}"'
            )
        else:
            print(f"{db_filename} ya existe.")

        return

    # ---------- flujo completo desde cero ------------------------------------
    if not args.skip_download:
        print("Descargando clips...")
        run_command(
            f'python3 download_clips.py --csv_file "{args.input}" '
            f'--searchterm "{args.searchterm}" '
            f'--output_dir "{out_dir}/videos/raw" --offset {args.offset}'
        )
    else:
        print("🔹 Skip download: se asume videos/raw listo.")

    # Procesamiento ASD
    in_raw = os.path.abspath(f"{out_dir}/videos/raw")
    out_masked = os.path.abspath(f"{out_dir}/videos/masked")
    print("Procesando con TalkNet‑ASD…")
    run_command(
        f'cd "{ASD_PATH}" && ./process_video.py '
        f'--input_dir "{in_raw}" --output_dir "{out_masked}" '
        f'--second {args.offset}'
    )

    # Filtrado a un hablante
    run_command(
        "python3 is_there_a_person_in_the_video.py "
        f'--videos_folder "{out_masked}" '
        f'--discarded_videos "{out_dir}/videos/discarded" '
        f'--matched_videos "{out_dir}/videos/1_person" '
        f'--check_hands {args.check_hands}'
    )

    # Argos y sucesivos
    run_argos_if_needed(out_dir)
    generate_parquet_if_needed(out_dir)

    # Análisis de voz en paralelo
    print("Ejecutando análisis de voz (ecos/apate)…")
    with concurrent.futures.ThreadPoolExecutor() as ex:
        fut_ecos = ex.submit(
            run_command, f'python3 ecos.py --input_folder "{out_dir}" --n_persons 1'
        )
        fut_apate = ex.submit(
            run_command, f'python3 apate.py --input_folder "{out_dir}" --n_persons 1'
        )
        concurrent.futures.wait([fut_ecos, fut_apate])

    if fut_ecos.result() == 0 and fut_apate.result() == 0:
        mark_apate_ecos_done(out_dir)
        print("ecos.py y apate.py completados correctamente.")
    else:
        print("⚠️  ecos/apate fallaron. Reintenta con --resume.")

    # Construcción del DuckDB
    run_command(
        f'python3 hefesto.py --input_folder "{out_dir}" '
        f'--n_persons 1 --db_name "{out_dir}/dataset.duckdb"'
    )

    # Copia de vídeos “buenos”
    good_dir = os.path.join(out_dir, "videos", "good")
    os.makedirs(good_dir, exist_ok=True)
    parquet_dir = os.path.join(out_dir, "dataset", "1_persons", "parquet_files")
    if os.path.exists(parquet_dir):
        for pq in os.listdir(parquet_dir):
            if pq.endswith(".parquet"):
                mp4 = pq.replace(".parquet", ".mp4")
                src = os.path.join(out_dir, "videos", "raw", mp4)
                if os.path.exists(src):
                    run_command(f'cp "{src}" "{good_dir}/{mp4}"')

if __name__ == "__main__":
    main()

