#!/usr/bin/env python3

import cv2
import sys
import argparse
from ultralytics import YOLO
import os
import shutil
from datetime import datetime
import torch


bodyPoints = { 'Nose': 0, 'LeftEye': 1, 'RightEye': 2, 'LeftEar': 3, 'RightEar': 4, 'LeftShoulder': 5, 
          'RightShoulder': 6, 'LeftElbow': 7, 'RightElbow': 8, 'LeftWrist': 9, 'RightWrist': 10, 
          'LeftHip': 11, 'RightHip': 12, 'LeftKnee': 13, 'RightKnee': 14, 'LeftAnkle': 15, 'RightAnkle': 16 }

def person_with_hands_in_image(image, model, hands):
  """
  Returns True if there is a person with visible head and shoulders in the image .
  """
  results = model(image, conf=0.65, verbose=False)
  
  datetimeinms = datetime.now().strftime('%Y%m%d%H%M%S%f')
  for result in results: 
    result.save(filename = "images/" + datetimeinms + ".jpg")

  try:
    num_people = len(results[0].keypoints)
    keypoints = results[0].keypoints
    #print("NUM_PEOPLE*******: " + str(num_people))

    if num_people != 1:
      # print("FRAME MALO, NUM_PEOPLE: " + str(num_people))
      return False, num_people
      
    else:
      for i in range(num_people):
        def is_zero_keypoint(keypoint: torch.Tensor) -> bool:
            """Check if a keypoint is [0., 0.]."""
            zero_tensor = torch.tensor([0., 0.], device=keypoint.device)
            return torch.all(keypoint.eq(zero_tensor)).item()

        nose = not is_zero_keypoint(keypoints.xy[i][bodyPoints['Nose']])
        right_shoulder = not is_zero_keypoint(keypoints.xy[i][bodyPoints['RightShoulder']])
        left_shoulder = not is_zero_keypoint(keypoints.xy[i][bodyPoints['LeftShoulder']])
        if hands:
          right_wrist = not is_zero_keypoint(keypoints.xy[i][bodyPoints['RightWrist']])
          left_wrist = not is_zero_keypoint(keypoints.xy[i][bodyPoints['LeftWrist']])
        else:
          right_wrist = True
          left_wrist = True 
        if nose and right_shoulder and left_shoulder and right_wrist and left_wrist:
          # print("FRAME BUENO")
          return True, 1
        else:
          # print("FRAME MALO")
          return False, 1
  except Exception as e:
    # print("EXCEPTION: " + str(e))
    return False, 0
  
# extract 15 frames from a given video separates in time intervals of video lenght / 16 
def extract_frames(video_path):
  """
  Extract 15 frames from a given video separates in time intervals of video lenght / 16 
  """
  cap = cv2.VideoCapture(video_path)
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  interval = int(length / 16)
  frames = []
  for i in range(15):
    cap.set(cv2.CAP_PROP_POS_FRAMES, (i+1) * interval)
    ret, frame = cap.read()
    if ret:
      frames.append(frame)
  cap.release()
  
  return frames

def determine_if_person_in_frames(frames, model, check_hands):
  """
  Returns True if there is a person with visible head and hands in any of the frames.
  """
  positive_count = 0
  max_people = 0
  for frame in frames:
    shoulders, num_people = person_with_hands_in_image(frame, model, check_hands)
    max_people = max(max_people, num_people)
    if shoulders:
      positive_count += 1
  if positive_count > 5:
    return True, max_people
  return False, max_people


# if not loaded as library execute main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Determine if there is a person with visible head and hands in a video.')
    #parser.add_argument('--video', type=str, required=False, help='Path to the video file.')
    parser.add_argument('--videos_folder', type=str, required=False, help='Path to the folder with input videos.')
    parser.add_argument('--discarded_videos', type=str, required=False, help='Path to the file with discarded videos.')
    parser.add_argument('--matched_videos', type=str, required=False, help='Path to the folder with matched videos.')
    parser.add_argument('--check_hands', type=bool, required=False, help='If True, the model will look for hands in the image.', default=False)

    args = parser.parse_args()

    # if videos_folder doesn't exist exit
    if not os.path.exists(args.videos_folder):
      print('Folder with videos doesn\'t exist.')
      exit()
    
    # if discarded_videos folder doesn't exist create it
    if not os.path.exists(args.discarded_videos):
      os.makedirs(args.discarded_videos)

    # if matched_videos folder doesn't exist create it
    if not os.path.exists(args.matched_videos):
      os.makedirs(args.matched_videos)



    model = YOLO("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt")
                 
    #yolo11x-pose.pt") 

    # loop over the .mp4 files recursively in the videos_folder
    for root, dirs, files in os.walk(args.videos_folder):
      for file in files:
        if file.endswith(".mp4"):
          video_path = os.path.join(root, file)
          frames = []
          frames = extract_frames(video_path)
          shoulders, num_people = determine_if_person_in_frames(frames, model, args.check_hands)
          # print video filename and number of people detected and if there are shoulders
          print("FILE: " + video_path + " NUM_PEOPLE: " + str(num_people) + " SHOULDERS AND HANDS: " + str(shoulders))
          if num_people != 1:
            subdir = os.path.join(args.discarded_videos, str(num_people))
            if not os.path.exists(subdir):
              os.makedirs(subdir)
            # copy video file to discarded_videos folder
            
            shutil.copy(video_path, subdir)
          elif shoulders:
            # copy the video to the matched_videos folder
            shutil.copy(video_path, args.matched_videos)
          else:
            # copy video file to discarded_videos/1 folder 
            if not os.path.exists(os.path.join(args.discarded_videos, '1')):
              os.makedirs(os.path.join(args.discarded_videos, '1'))
            shutil.copy(video_path, os.path.join(args.discarded_videos, '1'))

    # chmod 777 discarded_videos and matched_videos folders and files
    os.system('chmod -R 777 ' + args.discarded_videos)
    os.system('chmod -R 777 ' + args.matched_videos)
