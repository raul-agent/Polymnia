#!/usr/bin/env python3

import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run Parole prosody analysis on a folder of videos")
    parser.add_argument('--input_folder', help='Folder with videos and dataset', required=True)
    parser.add_argument('--n_persons', help='Number of persons in the videos', required=True, type=int, default=1)
    args = parser.parse_args()

    n_persons = f"{args.n_persons}_persons"
    parquet_dir = os.path.join(args.input_folder, "dataset", n_persons, "parquet_files")
    output_dir = os.path.join(args.input_folder, "dataset", n_persons, "parquet_speech_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Guardar el directorio actual
    original_cwd = os.getcwd()

    # Directorio donde vive parole.sh
    parole_dir = os.path.join(os.path.dirname(__file__), "create_datasets")

    for parquet_file in os.listdir(parquet_dir):
        if parquet_file.endswith(".parquet"):
            video_name = parquet_file.replace(".parquet", ".mp4")
            video_path = os.path.join(args.input_folder, "videos", "raw", video_name)

            if not os.path.exists(video_path):
                print(f"⚠️  Video not found for {parquet_file}, skipping.")
                continue

            print(f"🔧 Processing: {video_name}")

            try:
                # Entrar al directorio de parole y ejecutar el script
                subprocess.run(
                    ["./parole/scripts/parole.sh", video_path, output_dir, "--parquet"],
                    check=True,
                    cwd=parole_dir
                )
            except subprocess.CalledProcessError as e:
                print(f"❌ Parole failed on {video_name}: {e}")

    # Volver al directorio original
    os.chdir(original_cwd)

if __name__ == '__main__':
    main()

