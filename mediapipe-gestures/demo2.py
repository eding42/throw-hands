import mediapipe as mp
import cv2
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime
from pynput.keyboard import Key, Controller
import json

colors = ["red", "blue", "green"]
keyboard = Controller()

def getCurrentTime():
    current_date = datetime.now()
    dateInt = 0
    dateInt = current_date.year*10000000000 + current_date.month * 100000000 + current_date.day * 1000000 + current_date.hour*10000 + current_date.minute*100 + current_date.second

    return dateInt


# Define a function to handle the case where no gestures are detected
def handle_gestures(truth):
    if truth == True:
        colors.append("entry")
        list_length = len(colors)
        if list_length > 50:
            keyboard.press(Key.down)
            keyboard.release(Key.down)
            colors.clear()

    elif truth == False:
        colors.clear()
        print("No GESTURES DETECTED! ")
    return

model_path = 'C:/Users/eding/coding/mediapipe-gestures/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

capture = cv2.VideoCapture(0)
previousTime = 0
currentTime = 0

        
# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if not result.gestures:
        handle_gestures(False)
    else:
        toPrint = str(result.gestures[0][0].category_name)
        if toPrint != "None":
            handle_gestures(True)
            print(toPrint)
        # for current in result.gestures[0]:
        #     print(current)
        # print('gesture recognition result: {}'.format(result))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with GestureRecognizer.create_from_options(options) as recognizer:
    while capture.isOpened():
        now = capture.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = capture.read()
        frame_array = np.asarray(frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_array)
        recognition_result = recognizer.recognize_async(mp_image, int(now))


capture.release()
cv2.destroyAllWindows()        

