import mediapipe as mp
import cv2
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import asyncio
import websockets
import json



# Fetch the service account key JSON file contents
cred = credentials.Certificate('C:/Users/eding/coding/hand-location-recog/boxerbabylon-firebase-adminsdk-hyr4u-e17f53542d.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://boxerbabylon-default-rtdb.firebaseio.com/'
})

model_path = 'C:/Users/eding/coding/hand-location-recog/pose_landmarker_lite.task'

colors = ["red", "blue", "green"]

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

capture = cv2.VideoCapture(0)



def save_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    right_fist = ((float(result.pose_landmarks[0][20].x)+float(result.pose_landmarks[0][16].x))/2.0,(float(result.pose_landmarks[0][20].y)+float(result.pose_landmarks[0][16].y))/2.0,(float(result.pose_landmarks[0][22].z)))
    left_fist = ((float(result.pose_landmarks[0][19].x)+float(result.pose_landmarks[0][15].x))/2.0,(float(result.pose_landmarks[0][15].y)+float(result.pose_landmarks[0][19].y))/2.0,(float(result.pose_landmarks[0][19].z)))
    if (result.pose_landmarks):
        # right_firstAvgX = sum(moving_averageX)/len(moving_averageX)
        ref = db.reference("/Game/Boxer/")
        if(len(colors) > 5):
            data = {
                "Boxer": {
                    "LeftGlove": {
                        # "x" : 0.0,
                        "x": left_fist[0]*(-10.0)+6.0,
                        "y": left_fist[1]*(-10.0)+8.5,
                        # "y" : 0.0,
                        "z": left_fist[2]*(6.0) + 8.0
                    },
                    "RightGlove": {
                        # "x" : 0.0,
                        "x": right_fist[0]*(-10.0)+5.0,
                        # "y": 0.0,
                        "y": right_fist[1]*(-10.0)+8.5,
                        "z": right_fist[2]* (6.0) 
                    }
                }
            }
            print(right_fist)
            print(left_fist)
            json_data = json.dumps(data, indent=2)
            with open("C:/Users/eding/coding/BabylonBoxer/data.json", "w") as json_file:
                json_file.write(json_data)

            # ref.update({
            #     "LeftGlove/":
            #     {
            #         "x": left_fist[0]*(-10.0),
            #         "y": left_fist[1]*(-10.0),
            #         "z": left_fist[2]*(-10.0)
            #     },
            #     "RightGlove/":
            #     {
            #         "x": 0.0,
            #         "y": 0.0,
            #         "z": 0.0
            #     }
            # })
            # ref.update({
            #     "LeftGlove/":
            #     {
            #         "x": left_fist[0],
            #         "y": left_fist[1],
            #         "z": left_fist[2]
            #     },
            #     "RightGlove/":
            #     {
            #         "x": right_fist[0],
            #         "y": right_fist[1],
            #         "z": right_fist[2]
            #     }
            # })

            colors.clear()
        else:
            # print("skipping")
            colors.append("admissions")
        # print(right_fist)


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=save_result)

with PoseLandmarker.create_from_options(options) as landmarker:
    while capture.isOpened():
        now = capture.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_array = np.asarray(frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_array)
        recognition_result = landmarker.detect_async(mp_image, int(now))
