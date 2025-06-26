#!/usr/bin/env python3

import argparse
import os
import subprocess
import time
import logging

def download_clip_with_retries(line, searchterm, output_dir, offset, max_retries=3):
    try:
        parts = line.strip().split('\t')
        if len(parts) < 6:
            logging.error(f"Línea inválida: {line}")
            return False

        # Extraer datos y formatear el nombre
        name_raw = parts[0]
        name = os.path.basename(name_raw.replace('.txt', ''))
        clip_id = parts[1]
        start_sec = int(parts[2])
        start_msec = int(parts[3])
        end_sec = int(parts[4])
        end_msec = int(parts[5])

        # Usamos divisor 1000 si el nombre empieza por "2017", sino 100
        divisor = 1000 if name.startswith("2017") else 100
        start = round(start_sec + start_msec / divisor - float(offset), 3)
        end = round(end_sec + end_msec / divisor + float(offset), 3)

        filename = f"{name}_{start}_{end}_{searchterm}.mp4"
        output_path = os.path.join(output_dir, filename)

        url = f"https://gallo.case.edu/cgi-bin/snippets/newsscape_mp4_snippet.cgi?file={clip_id}&start={start}&end={end}"

        # Comando curl envuelto en timeout (20s)
        cmd = [
            "timeout", "20",
            "curl",
            "--connect-timeout", "5",
            "--max-time", "15",
            "-L",
            "-A", "Mozilla/5.0",  # Cabecera tipo navegador
            "-o", output_path,
            url
        ]

        attempt = 1
        while attempt <= max_retries:
            print(f"[INFO] Descargando {filename} (intento {attempt}/{max_retries})")
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                print(f"[OK] Descargado: {filename}")
                return True
            else:
                print(f"[WARNING] Fallo en {filename} (intento {attempt})")
                logging.warning(f"Fallo en {filename} (intento {attempt}). Código: {result.returncode}")
                attempt += 1
                time.sleep(5)  # Pausa entre reintentos

        logging.error(f"Descarga fallida tras {max_retries} intentos: {filename}")
        return False

    except Exception as e:
        logging.exception(f"Excepción en la descarga de {line}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Descarga clips de gallo.case.edu de forma secuencial con reintentos y pausas."
    )
    parser.add_argument("--csv_file", required=True, help="Archivo CSV con los clips a descargar")
    parser.add_argument("--searchterm", required=True, help="Término de búsqueda para agregar al nombre (alfanumérico o con '_')")
    parser.add_argument("--offset", type=float, default=2.5, help="Offset a agregar a los tiempos de inicio y fin")
    parser.add_argument("--output_dir", required=True, help="Directorio de salida para los clips")
    args = parser.parse_args()

    # Validación básica para searchterm: permite alfanumérico y guiones bajos
    if not args.searchterm.replace("_", "").isalnum():
        print("El término de búsqueda debe ser alfanumérico o contener solo '_'")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Configurar logging en un archivo para fallos
    logging.basicConfig(
        filename="failed_downloads.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    with open(args.csv_file, "r") as f:
        lines = f.readlines()

    total = len(lines)
    print(f"[INFO] Se procesarán {total} clips.")
    count = 0

    for line in lines:
        count += 1
        print(f"[INFO] Procesando clip {count}/{total}")
        download_clip_with_retries(line, args.searchterm, args.output_dir, args.offset)
        time.sleep(2)  # Pausa de 2 segundos entre cada descarga

    print("[INFO] Proceso finalizado.")

if __name__ == "__main__":
    main()

