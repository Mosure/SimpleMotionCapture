import cv2
import numpy as np
import datetime
import os
import math


# Set our CWD so we can get proper dlls
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Clear console
os.system('cls')


# Configuration Variables
output_dir = "X:/Security/"
record_duration = 180
startup_frames = 120
framerate = 30
detection_interval = 10
area_threshold = 8000
debug = False

# Video Input
vcap = cv2.VideoCapture("rtsp://ipc.home.lan:554/live/ch0")

# Kernel definitions (define these only once)
kernel_small = np.ones((9, 9), np.uint8)
kernel_large = np.ones((51, 51), np.uint8)

# These variables are used between frames
frame = 0
cont_sorted = []
total_area = 0
frames_left = 0
total_frames = 0

# Our VideoWriter object (used between frames)
vwrite = {}

# Background change detector (used between frames)
fgbg = cv2.createBackgroundSubtractorMOG2()


while vcap.isOpened():
    ret, f = vcap.read()
    frame += 1

    if ret:
        ## Draw rectangles around changed areas (updates every 'detection_interval' frames)
        #for i in range(len(cont_sorted)):
        #    x, y, w, h = cv2.boundingRect(cont_sorted[i])
        #    cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Debug the video feed
        if debug:
            cv2.imshow('Video', f)

        # Give the background change detector some time to startup
        if startup_frames > 0:
            startup_frames -= 1

        # Apply the new frame to the background change detector
        fg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        fmask = fgbg.apply(fg)

        # If we are recording, add the frame to the video writer output
        if frames_left > 0:
            vwrite.write(f)

            total_frames += 1
            frames_left -= 1

            # If the video writer is finished, close the video writer and log the duration
            if frames_left == 0:
                vwrite.release()
                print("[RECORDING] Finished: " + str(math.ceil(total_frames / framerate)) + " seconds")
                total_frames = 0

        # If we are not in the startup period, and the frame lands on a detection_interval frame...
        if startup_frames == 0 and frame % detection_interval == 0:

            # Make the background detection pixels much larger
            fmask = cv2.morphologyEx(fmask, cv2.MORPH_OPEN, kernel_small)

            # Make sure they are all the same color
            ret, fmask = cv2.threshold(fmask, 200, 255, 0)

            # Make the pixels EVEN larger (we are trying to make them into a large glob)
            fmask = cv2.dilate(fmask, kernel_large, iterations=1)

            # Find the contours of the large glob
            contours, hierarchy = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            
            # Make a bounding box around the large glob
            total_area = 0
            for i in range(len(cont_sorted)):
                x, y, w, h = cv2.boundingRect(cont_sorted[i])

                # Add the bounding box area to our total area of movement
                total_area += w * h

            # If the total area is above our threshold, start recording
            if total_area > area_threshold:
                if frames_left == 0:
                    # Create a new file by datetime
                    filename = output_dir + datetime.datetime.now().strftime("%Y-%m-%d %HH %MM %SS") + '.avi'

                    vwrite = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'h264'), framerate, (f.shape[1], f.shape[0]))

                    print("[RECORDING] Generating file: " + filename)
                    
                frames_left = record_duration

    # Listen for q to be pressed (closes the program)
    if debug:
        kbIn = cv2.waitKey(1) & 0xFF
        if kbIn == ord('q'):
            break

# Close windows nicely (only if we show them)
if debug:
    cv2.destroyAllWindows()
