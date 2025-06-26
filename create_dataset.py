#!/usr/bin/env python3
import argparse
import os
import concurrent.futures
import sys

# Obtener la ruta del script Python
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definir la variable de entorno para el script R
os.environ["PYTHON_SCRIPT_DIR"] = script_dir

ASD_PATH = "/home/brian/TalkNet-ASD"

def run_command(command):
    """
    Ejecuta un comando con os.system (retorna el código de salida).
    """
    print(f"Ejecutando: {command}")
    return os.system(command)

def check_json_generated(output_folder):
    """
    Comprueba si en output_folder/dataset/OpenPose existen carpetas con JSON_FILES.
    """
    openpose_dir = os.path.join(output_folder, "dataset", "OpenPose")
    if not os.path.exists(openpose_dir):
        return False
    for video_name in os.listdir(openpose_dir):
        video_dir = os.path.join(openpose_dir, video_name)
        if os.path.isdir(video_dir):
            json_folder = os.path.join(video_dir, "JSON_FILES")
            if os.path.exists(json_folder):
                return True
    return False

def check_parquet_generated(output_folder):
    """
    Comprueba si existen archivos .parquet en output_folder/dataset/1_persons/parquet_files.
    """
    parquet_dir = os.path.join(output_folder, "dataset", "1_persons", "parquet_files")
    if os.path.exists(parquet_dir) and any(f.endswith('.parquet') for f in os.listdir(parquet_dir)):
        return True
    return False

def check_apate_ecos_done(output_folder):
    """
    Comprueba la existencia de archivos bandera que indican que se completaron ecos.py y apate.py.
    """
    apate_flag = os.path.join(output_folder, "apate_done.txt")
    ecos_flag = os.path.join(output_folder, "ecos_done.txt")
    return os.path.exists(apate_flag) and os.path.exists(ecos_flag)

def mark_apate_ecos_done(output_folder):
    """
    Marca que ambos (apate y ecos) se completaron con éxito.
    Esto se usa cuando se ejecutan en el modo completo (no-resume) y terminan bien.
    """
    with open(os.path.join(output_folder, "apate_done.txt"), "w") as f:
        f.write("done")
    with open(os.path.join(output_folder, "ecos_done.txt"), "w") as f:
        f.write("done")

def run_missing_scripts(output_folder):
    """
    En modo recuperación, si alguno de los dos scripts (apate.py o ecos.py)
    no tiene su flag 'done', se lanza SOLO ese script.
    Si faltan ambos, se lanzan en paralelo.

    - Si un script falla (código de salida != 0), NO se crea su flag.
    - Si termina bien, se crea <nombre_script>_done.txt
    """
    apate_flag = os.path.join(output_folder, "apate_done.txt")
    ecos_flag = os.path.join(output_folder, "ecos_done.txt")

    tasks = []
    if not os.path.exists(apate_flag):
        tasks.append(("apate", f'python3 apate.py --input_folder {output_folder} --n_persons 1', apate_flag))
    if not os.path.exists(ecos_flag):
        tasks.append(("ecos", f'python3 ecos.py --input_folder {output_folder} --n_persons 1', ecos_flag))

    if not tasks:
        print("No se necesitan ejecutar apate.py ni ecos.py (ambos flags presentes).")
        return
    
    # Ejecutamos todas las tareas que faltan (1 o 2) en paralelo.
    print("Ejecutando scripts faltantes en modo recuperación...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_task = {}
        for (name, cmd, flag_path) in tasks:
            future = executor.submit(run_command, cmd)
            future_to_task[future] = (name, flag_path)

        for future in concurrent.futures.as_completed(future_to_task):
            name, flag_path = future_to_task[future]
            retcode = future.result()  # Código de salida devuelto por os.system
            if retcode == 0:
                # Éxito
                print(f"{name}.py se ejecutó correctamente. Marcando como completado.")
                with open(flag_path, "w") as f:
                    f.write("done")
            else:
                print(f"⚠️  {name}.py terminó con un error (código {retcode}). No se marcará como completado.")

def run_recovery(output_folder, offset, check_hands):
    """
    Modo recuperación: Reanuda el proceso en los pasos faltantes,
    basándose en la existencia de JSON, parquet y archivos bandera.
    """
    dataset_dir = os.path.join(output_folder, "dataset")
    
    # Paso 1: Si existen JSON pero no los parquet, ejecuta Rscript para cada video.
    if check_json_generated(output_folder) and not check_parquet_generated(output_folder):
        print("JSON encontrados pero no se han generado los parquet. Ejecutando Rscript max_people_classification.R...")
        openpose_dir = os.path.join(dataset_dir, "OpenPose")
        for video_name in os.listdir(openpose_dir):
            video_path = os.path.join(openpose_dir, video_name)
            json_folder = os.path.join(video_path, "JSON_FILES")
            if os.path.exists(json_folder):
                print(f"Ejecutando max_people_classification.R en {json_folder}")
                os.system(f'Rscript max_people_classification.R "{json_folder}" "{dataset_dir}" > /dev/null 2>&1')
    else:
        print("No es necesario ejecutar Rscript (JSON o parquet ya existentes).")
    
    # Paso 2: Si existen los parquet, revisar si falta apate.py o ecos.py
    if check_parquet_generated(output_folder):
        run_missing_scripts(output_folder)
    else:
        print("Aún no hay archivos parquet, no podemos ejecutar apate/ecos.")
    
    # Paso 3: Si no se ha generado el archivo duckdb, preguntar al usuario si lo quiere generar.
    duckdb_file = os.path.join(output_folder, "dataset.duckdb")
    if not os.path.exists(duckdb_file):
        user_input = input("Todos los outputs están presentes. ¿Deseas generar el archivo duckdb? (s/n): ")
        if user_input.lower() == "s":
            print("Generando duckdb...")
            os.system(f'python3 hefesto.py --input_folder {output_folder} --n_persons 1 --db_name "{duckdb_file}"')
        else:
            print("Generación de duckdb cancelada por el usuario.")
    else:
        print("El archivo duckdb ya existe.")

def main():
    parser = argparse.ArgumentParser(description="Ejecuta todas las herramientas para crear un dataset multimodal.")
    parser.add_argument('--input', help='Archivo csv generado por tabulate', required=True)
    parser.add_argument('--output_folder', help='Carpeta de salida para guardar el dataset', required=True)
    parser.add_argument('--searchterm', help='Término de búsqueda para descargar los videos', required=True)
    parser.add_argument('--offset', help='Offset para descargar los videos', required=True, type=int, default=2)
    parser.add_argument('--check_hands', help='Comprobar manos en los videos', required=False, type=bool, default=False)
    parser.add_argument('--resume', help='Modo recuperación: reanuda el proceso con outputs ya existentes', action='store_true')
    
    args = parser.parse_args()
    searchterm = args.searchterm
    offset = args.offset

    if not os.path.exists(args.input):
        print("El archivo de entrada no existe.")
        return
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # Si se activa el modo recuperación, se chequean los pasos ya completados y se ejecutan los que faltan.
    if args.resume:
        print("Modo recuperación activado.")
        run_recovery(args.output_folder, offset, args.check_hands)
        return  # Salir tras la recuperación
    
    # Si no se activa el modo recuperación, se ejecuta el proceso completo desde el inicio:
    print("Descargando clips...")
    os.system(
        f'python3 download_clips.py '
        f'--csv_file {args.input} '
        f'--searchterm {searchterm} '
        f'--output_dir {args.output_folder}/videos/raw '
        f'--offset {offset}'
    )
   
    abs_ASD_input_dir = os.path.abspath(f"{args.output_folder}/videos/raw")
    abs_ASD_output_dir = os.path.abspath(f"{args.output_folder}/videos/masked")

    print(f"ASD_input_dir: {abs_ASD_input_dir}")
    print(f"ASD_output_dir: {abs_ASD_output_dir}")
    
    print("Procesando videos con TalkNet-ASD...")
    current_path = os.getcwd()
    os.system(
        f'cd {ASD_PATH} && '
        f'./process_video.py --input_dir "{abs_ASD_input_dir}" '
        f'--output_dir "{abs_ASD_output_dir}" '
        f'--second {offset}'
    )
    os.chdir(current_path)

    print("Descartando clips con más de una persona...")
    os.system(
        f'python3 is_there_a_person_in_the_video.py '
        f'--videos_folder {args.output_folder}/videos/masked '
        f'--discarded_videos {args.output_folder}/videos/discarded '
        f'--matched_videos {args.output_folder}/videos/1_person '
        f'--check_hands {args.check_hands}'
    )
   
    print("Ejecutando OpenPose y dfMaker... (esto puede tardar un poco)")
    os.system(
        f'python3 argos.py '
        f'--videos_folder {args.output_folder}/videos/1_person '
        f'--output_folder {args.output_folder} '
        f'--openpose_path /opt/openpose '
        f'--face_hands True '
        f'--skeletons True'
    )
    
    print("Ejecutando análisis de voz (ecos.py y apate.py) en paralelo (modo completo)...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_ecos = executor.submit(run_command, f'python3 ecos.py --input_folder {args.output_folder} --n_persons 1')
        future_apate = executor.submit(run_command, f'python3 apate.py --input_folder {args.output_folder} --n_persons 1')
        concurrent.futures.wait([future_ecos, future_apate])

    # Si ambos terminaron con exit code 0, marcamos como done.
    # (Aquí asumimos que, en la mayoría de casos, terminan bien. 
    #  Si uno falló, tu "modo no-recovery" no marca, y podrás retomar con --resume).
    if (future_ecos.result() == 0) and (future_apate.result() == 0):
        mark_apate_ecos_done(args.output_folder)
        print("Ambos scripts de análisis de voz completados con éxito.")
    else:
        print("⚠️  Al menos uno de los scripts (ecos/apate) falló. Usa --resume para reintentarlo.")

    print("Combinando todos los datos y generando el dataset...")
    os.system(
        f'python3 hefesto.py '
        f'--input_folder {args.output_folder} '
        f'--n_persons 1 '
        f'--db_name {args.output_folder}/dataset.duckdb'
    )

    # Mover los videos buenos a una carpeta
    os.makedirs(os.path.join(args.output_folder, 'videos', 'good'), exist_ok=True)
    parquet_folder = os.path.join(args.output_folder, 'dataset', '1_persons', 'parquet_files')
    if os.path.exists(parquet_folder):
        for parquet_file in os.listdir(parquet_folder):
            if parquet_file.endswith('.parquet'):
                video_name = parquet_file.replace('.parquet', '.mp4')
                src_video_path = os.path.join(args.output_folder, "videos", "raw", video_name)
                dest_video_path = os.path.join(args.output_folder, "videos", "good", video_name)
                if os.path.exists(src_video_path):
                    os.system(f'cp "{src_video_path}" "{dest_video_path}"')

if __name__ == '__main__':
    main()

