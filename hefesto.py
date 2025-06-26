#! /usr/bin/env python3

import os
from typing import Dict, List, Any
import pandas as pd
import pyarrow.parquet as pq
import json
import duckdb
import argparse
from datetime import datetime, timezone

# Contadores globales para discrepancias
discrepancy_count = 0
discrepancy_files = []

def prepare_prosody_data(prosody_path: str) -> pd.DataFrame:
    """
    Load and prepare prosody data from a parquet file.
    """
    try:
        prosody_df = pq.read_table(prosody_path).to_pandas()
    except Exception as e:
        print(f"[ERROR] Reading prosody file {prosody_path}: {e}")
        return pd.DataFrame()
    
    prosody_reduced_df = prosody_df.iloc[::6].rename(columns={"Frame": "frame"})
    prosody_reduced_df['frame'] = prosody_reduced_df['frame'].astype(int)
    
    # Convert datetime to UTC, then remove timezone information
    if 'datetime' in prosody_reduced_df.columns:
        if prosody_reduced_df['datetime'].dt.tz:
            prosody_reduced_df['datetime'] = prosody_reduced_df['datetime'].dt.tz_convert(timezone.utc)
        prosody_reduced_df['datetime'] = prosody_reduced_df['datetime'].dt.tz_localize(None)
        
        # Ensure microsecond precision
        prosody_reduced_df['datetime'] = pd.to_datetime(prosody_reduced_df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    
    return prosody_reduced_df

def prepare_words_data(words_path: str) -> pd.DataFrame:
    """
    Load and prepare words data from a JSON file.
    """
    if not os.path.exists(words_path):
        return pd.DataFrame({
            "frame": [0],
            "words": [""],
            "scores_w": [""]
        })
    
    try:
        with open(words_path, 'r') as f:
            words = json.load(f)
        
        processed_words = []
        for frame in words:
            processed_words.append({
                "frame": frame["frame_number"],
                "words": ", ".join(frame["words"]),
                "scores_w": ", ".join(map(str, frame["scores"]))
            })
        
        return pd.DataFrame(processed_words)
    except Exception as e:
        print(f"[ERROR] Processing words file {words_path}: {e}")
        return pd.DataFrame({
            "frame": [0],
            "words": [""],
            "scores_w": [""]
        })

def prepare_pose_data(pose_path: str) -> pd.DataFrame:
    """
    Load and prepare pose data from a parquet file.
    """
    try:
        pose_df = pq.read_table(pose_path).to_pandas()
        pose_df["frame"] += 1
        return pose_df
    except Exception as e:
        print(f"[ERROR] Reading pose file {pose_path}: {e}")
        return pd.DataFrame()

def combine_all_data(prosody_data: pd.DataFrame, words_data: pd.DataFrame, pose_data: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Combine prosody, words, and pose data into a single DataFrame.
    """
    if len(prosody_data) != len(words_data):
       print(f"[WARNING] Mismatch in row numbers for file: {filename} in speech analysis and words data")
    
    combined_prosody_words = prosody_data.merge(words_data, on='frame', how='outer')
    final_combined_data = pd.merge(combined_prosody_words, pose_data, on="frame", how="outer")
    
    # Ensure datetime is in correct format
    if 'datetime' in final_combined_data.columns:
        final_combined_data['datetime'] = pd.to_datetime(final_combined_data['datetime'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    
    return final_combined_data

def process_single_file(parquet_file_path: str, db_file: str) -> None:
    """
    Process a single pose file and related files.
    """
    global discrepancy_count, discrepancy_files
    print(f"Processing file: {parquet_file_path}")

    # Derivar rutas de archivos relacionados
    prosody_file_path = parquet_file_path.replace("parquet_files", "parquet_speech_analysis")
    words_file_path = parquet_file_path.replace("parquet_files", "whisperx_framer_output").replace(".parquet", ".json")
    
    # Verificar existencia del archivo de prosodia
    if not os.path.exists(prosody_file_path):
        print(f"[WARNING] Prosody file not found: {prosody_file_path}")
        discrepancy_count += 1
        discrepancy_files.append(prosody_file_path)
        return
    
    prosody_data = prepare_prosody_data(prosody_file_path)
    if prosody_data.empty:
        print(f"[WARNING] Prosody data is empty for file: {prosody_file_path}")
        discrepancy_count += 1
        discrepancy_files.append(prosody_file_path)
        return

    words_data = prepare_words_data(words_file_path)
    pose_data = prepare_pose_data(parquet_file_path)
    combined_data = combine_all_data(prosody_data, words_data, pose_data, parquet_file_path)
    
    if combined_data.empty:
        print(f"[WARNING] Combined data is empty for file: {parquet_file_path}")
        discrepancy_count += 1
        discrepancy_files.append(parquet_file_path)
        return
    
    try:
        con = duckdb.connect(db_file)
        con.register("combined_data", combined_data)
        con.execute("CREATE TABLE IF NOT EXISTS multi_data AS SELECT * FROM combined_data LIMIT 0")
        con.execute("INSERT INTO multi_data SELECT * FROM combined_data")
        # Eliminar filas donde id es null
        con.execute("DELETE FROM multi_data WHERE id is null")
        con.close()
        print(f"File processed and inserted into DB: {parquet_file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to insert data from {parquet_file_path} into DuckDB: {e}")
        discrepancy_count += 1
        discrepancy_files.append(parquet_file_path)

def main() -> None:
    """
    Main function to parse command line arguments and process files.
    """
    parser = argparse.ArgumentParser(description="Process multimodal data files.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to the input folder")
    parser.add_argument("-n", "--n_persons", type=str, required=True, help="Number of persons")
    parser.add_argument("-d", "--db_name", type=str, default="minos.duckdb", help="Path to DuckDB database file")
    args = parser.parse_args()
    
    n_persons = f"{args.n_persons}_persons"
    parquet_dir = os.path.join(args.input_folder, "dataset", n_persons, "parquet_files")

    if not os.path.exists(parquet_dir):
        print(f"[ERROR] Parquet folder does not exist: {parquet_dir}")
        return

    print(f"Input folder: {args.input_folder}")
    print(f"Number of persons: {args.n_persons}")
    print(f"DuckDB database file: {args.db_name}")
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    print(f"Number of Parquet files found: {len(parquet_files)}")

    for parquet_file in parquet_files:
        file_path = os.path.join(parquet_dir, parquet_file)
        process_single_file(file_path, args.db_name)
    
    # Imprimir resumen final de discrepancias
    print(f"\nTotal discrepancies: {discrepancy_count}")
    if discrepancy_files:
        print("Files with issues:")
        for f in discrepancy_files:
            print(f)

if __name__ == "__main__":
    main()

