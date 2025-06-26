#!/usr/bin/env python3

import argparse
import os
import cv2
import json
from typing import Dict, List, Tuple

def process_video_and_json(video_path: str, json_path: str, output_dir: str) -> None:
    """
    Process a video file and its corresponding JSON file to extract frame timestamps and word segments.

    Args:
        video_path (str): Path to the video file.
        json_path (str): Path to the JSON file containing word segments.
        output_dir (str): Directory to save the output JSON file.

    Returns:
        None
    """
    # Extract frame timestamps
    frame_timestamps = frames_from_video(video_path)

    # Load and process JSON data
    with open(json_path, 'r') as file:
        data = json.load(file)

    word_segments = data['word_segments']
    processed_data = [
        {
            'words': segment['word'],
            'start': segment['start'],
            'end': segment['end'],
            'score': segment['score']
        }
        for segment in word_segments
        if all(key in segment for key in ['word', 'start', 'end', 'score'])
    ]

    # Process frames and words
    all_frames_data = []
    for frame_number, frame_time in frame_timestamps.items():
        words_in_frame = []
        scores_in_frame = []
        for word_info in processed_data:
            word_start, word_end = word_info['start'], word_info['end']
            if word_start <= frame_time[1] and word_end >= frame_time[0]:
                words_in_frame.append(word_info['words'])
                scores_in_frame.append(word_info['score'])

        frame_data = {
            'frame_number': frame_number,
            'words': words_in_frame,
            'scores': scores_in_frame
        }
        all_frames_data.append(frame_data)

    # Save processed data to JSON file
    json_file_name = os.path.basename(json_path)
    json_file_path = os.path.join(output_dir, json_file_name)
    with open(json_file_path, 'w') as file:
        json.dump(all_frames_data, file, indent=4)

    print(f"Data saved to {json_file_path}")

def frames_from_video(video_path: str) -> Dict[int, List[float]]:
    """
    Extract frame timestamps from a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        Dict[int, List[float]]: A dictionary mapping frame numbers to timestamp intervals.
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    success, _ = video.read()
    count = 0
    frames = []
    while success:
        count += 1
        frames.append(count / fps)
        success, _ = video.read()

    frame_timestamp = {}
    for index in range(1, len(frames) + 1):
        if index == 1:
            frame_timestamp[index] = [0, frames[index - 1]]
        else:
            frame_timestamp[index] = [frames[index - 2], frames[index - 1]]
    return frame_timestamp

def main():
    parser = argparse.ArgumentParser(description="Run speech analysis on a folder of videos")
    parser.add_argument('--input_folder', help='Folder with videos and dataset', required=True)
    parser.add_argument('--n_persons', help='number of persons in the videos', required=True, type=int, default=1)
    args = parser.parse_args()

    n_persons = f"{args.n_persons}_persons"
    
   # SEPARA la construcción del path
    parquet_dir = os.path.join(args.input_folder, "dataset", n_persons, "parquet_files")

    if not os.path.exists(parquet_dir):
        print(f"No parquet folder found at {parquet_dir}")
        return

    # Y haz el print de forma sencilla
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    print(f"Number of parquet files: {len(parquet_files)}")

    for parquet_file in os.listdir(os.path.join(args.input_folder, "dataset", n_persons, "parquet_files")):
        try:

            if parquet_file.endswith(".parquet"):
                video_name = parquet_file.replace(".parquet", ".mp4")
                video_path = os.path.join(args.input_folder, "videos", "raw", video_name)
                
                print(f"Processing {video_path}")
                # Create output folders
                whisperx_output_dir = os.path.join(args.input_folder, "dataset", n_persons, "whisperx", video_name)
                os.makedirs(whisperx_output_dir, exist_ok=True)
                # ffmpeg -y -hide_banner -loglevel error -i INPUTVIDEO.mp4 \
                #  -c:v copy -af loudnorm=I=-23:LRA=7:TP=-2.0:measured_I=-17.31:measured_LRA=5.71:measured_TP=-1.35 output.mp3
                audio_file = os.path.join(whisperx_output_dir, video_name.replace('.mp4', '.mp3'))
                print(audio_file)
                os.system(f"ffmpeg -y -hide_banner -loglevel error -i {video_path} -c:v copy -af loudnorm=I=-23:LRA=7:TP=-2.0:measured_I=-17.31:measured_LRA=5.71:measured_TP=-1.35 {audio_file}")
                # Run whisperx on the video
                os.system(f"whisperx --model large --output_dir {whisperx_output_dir} {audio_file}")

                # Run framer logic on the video
                json_path = os.path.join(whisperx_output_dir, video_name.replace(".mp4", ".json") )
                framer_output_dir = os.path.join(args.input_folder, "dataset", n_persons, "whisperx_framer_output")
                os.makedirs(framer_output_dir, exist_ok=True)
                process_video_and_json(video_path, json_path, framer_output_dir)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

if __name__ == '__main__':
    main()

