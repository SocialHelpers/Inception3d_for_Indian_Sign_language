"""
This script is working demo of our project
"""

import numpy as np
import cv2

from timer import Timer
from framesandoflow import frames_downsample, images_crop, frames2flows
from videocapture import video_start, frame_show, video_show, video_capture
from datagenerator import VideoClasses
from model_i3d import I3D_load
from predict import probability2label


def livedemo():
    fDurationAvg = 3.0  # seconds

    # files
    sClassFile = "class_ISL.csv"

    print("\nStarting gesture recognition live demo ... ")
    # load label description
    oClasses = VideoClasses(sClassFile)

    sModelFile = "model/20190322-1841-ISL105-oflow-i3d-top-best.h5"

    h, w = 224, 224
    keI3D = I3D_load(sModelFile, 40, (h, w, 2), oClasses.nClasses)
    if (keI3D):
        print("Model loaded successfully")
    # open a pointer to the webcam video stream
    oStream = video_start(device=0, tuResolution=(320, 240), nFramePerSecond=10)

    nCount = 0
    sResults = ""
    timer = Timer()

    # loop over action states
    while True:
        # show live video and wait for key stroke
        key = video_show(oStream, "green", "Press key to start", sResults, tuRectangle=(h, w))

        # start!
        if (key == ord('3') or key == ord('5')):
            # countdown n sec
            video_show(oStream, "orange", "Recording starts in ", tuRectangle=(h, w), nCountdown=3)

            # record video for n sec
            if key == ord('3'):
                fDurationAvg = 3
                fElapsed, arFrames, _ = video_capture(oStream, "red", "Recording ",
                                                      tuRectangle=(h, w), nTimeDuration=int(fDurationAvg),
                                                      bOpticalFlow=False)
            else:
                fDurationAvg = 5
                fElapsed, arFrames, _ = video_capture(oStream, "red", "Recording ",
                                                      tuRectangle=(h, w), nTimeDuration=int(fDurationAvg),
                                                      bOpticalFlow=False)
            print(
                "\nCaptured video: %.1f sec, %s, %.1f fps" % (fElapsed, str(arFrames.shape), len(arFrames) / fElapsed))

            # show orange wait box
            frame_show(oStream, "orange", "Translating sign ...", tuRectangle=(h, w))

            # crop and downsample frames
            arFrames = images_crop(arFrames, h, w)
            arFrames = frames_downsample(arFrames, 40)

            # Translate frames to flows - these are already scaled between [-1.0, 1.0]
            print("Calculate optical flow on %d frames ..." % len(arFrames))
            timer.start()
            arFlows = frames2flows(arFrames, bThirdChannel=False, bShow=True)
            print("Optical flow per frame: %.3f" % (timer.stop() / len(arFrames)))

            # predict video from flows
            print("Predict video with %s ..." % (keI3D.name))
            arX = np.expand_dims(arFlows, axis=0)
            arProbas = keI3D.predict(arX, verbose=1)[0]
            nLabel, sLabel, fProba = probability2label(arProbas, oClasses, nTop=3)

            sResults = "Sign: %s (%.0f%%)" % (sLabel, fProba * 100.)
            print(sResults)
            nCount += 1

        # quit
        elif key == ord('q'):
            break

    oStream.release()
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    livedemo()
