#!/usr/bin/env python3  

import argparse
import os


# openpose_process runs openpose on a video and saves the results to the output_folder/dataset/OpenPose/video_name
# it also runs max_people_classification.R on the json_folder and saves the results to the output_folder/dataset/video_name 
# folder structure:
# output_folder/
#   dataset/
#     OpenPose/
#       video_name/
#         JSON_FILES/
#         skeletons/
#     1_persons/
#       video_name/
#         parquet_files/
#     2_persons/
#       video_name/
#         parquet_files/
#     ...
def openpose_process(video, output_folder, openpose_path, face_hands, skeletons):
    openpose_bin = f'{openpose_path}/build/examples/openpose/openpose.bin'    
    video_name = video.split('/')[-1].replace('.mp4', '')
    # create the output folder for the video if it doesn't exist with term
    os.makedirs(f'{output_folder}/dataset/OpenPose/{video_name}/JSON_FILES', exist_ok=True)

    # if skeletons is true create the skeletons folder
    if skeletons:
        skeletons_folder = f'{output_folder}/dataset/OpenPose/{video_name}/skeletons'
        if not os.path.exists(skeletons_folder):
            os.makedirs(skeletons_folder)

    # save the current path
    current_path = os.getcwd()
    # cd into the openpose directory
    os.chdir(openpose_path)
    # run openpose on the video with the correct flags
    if face_hands and skeletons:
        os.system(f'{openpose_bin} --video {video} --write_json {output_folder}/dataset/OpenPose/{video_name}/JSON_FILES --write_video {skeletons_folder}/skeleton.avi --display 0 --render_pose 1 --face --hand')
    elif face_hands:
        os.system(f'{openpose_bin} --video {video} --write_json {output_folder}/dataset/OpenPose/{video_name}/JSON_FILES --display 0 --render_pose 0 --face --hand')
    elif skeletons:
        os.system(f'{openpose_bin} --video {video} --write_json {output_folder}/dataset/OpenPose/{video_name}/JSON_FILES --write_video {skeletons_folder}/skeleton.avi --display 0 --render_pose 1')
    else:
        os.system(f'{openpose_bin} --video {video} --write_json {output_folder}/dataset/OpenPose/{video_name}/JSON_FILES --display 0 --render_pose 0')
        
    # cd back to the original directory
    os.chdir(current_path)

    # Call max_people_classification.R
    json_folder = f'{output_folder}/dataset/OpenPose/{video_name}/JSON_FILES'
    output_folder = f'{output_folder}/dataset'
    # output to log to /dev/null
    print(f'Running max_people_classification.R on {json_folder}')
    os.system(f'Rscript max_people_classification.R "{json_folder}" "{output_folder}" > /dev/null 2>&1')



def main():
    parser = argparse.ArgumentParser(description="Run OpenPose and dfMaker on a folder of videos. Video names must be in the format: 2017-07-21_0000_US_KABC_Eyewitness_News_5PM_356.505_361.531_before.mp4")
    parser.add_argument('--videos_folder', help='folder with videos to run OpenPose and dfMaker on', required=True)
    parser.add_argument('--output_folder', help='output folder to save the dataframes', required=True)
    parser.add_argument('--openpose_path', help='path to OpenPose', default='/opt/openpose')
    parser.add_argument('--face_hands', help='run OpenPose with face and hands', default=True)
    parser.add_argument('--skeletons', help='run OpenPose with skeletons', default=True)
    args = parser.parse_args()

    # if the videos_folder doesn't exist create it
    if not os.path.exists(args.videos_folder):
        print('Folder with videos doesn\'t exist.')
        return

    # if the output_folder doesn't exist create it
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    # loop through all the videos in the videos_folder
    for video in os.listdir(args.videos_folder):
        # check if the video is an mp4
        if video.endswith('.mp4'):
            # get the absolute path of the video
            full_video_path = os.path.abspath(f'{args.videos_folder}/{video}')
            full_output_path = os.path.abspath(args.output_folder)
            print(f'Processing {full_video_path}')
            openpose_process(f'{full_video_path}', full_output_path, args.openpose_path, args.face_hands, args.skeletons)

if __name__ == '__main__':
    main()
