import os
import shutil
import numpy as np
import pandas as pd
import warnings
from imutils import paths
import cv2


def create_folders(sPath):
    cnt = 1
    os.mkdir("Frames")
    FrameClassPath = sPath + "\Frames"
    os.chdir(FrameClassPath)
    sVideoPath=sPath+"\Classes"
    li_videos=os.listdir(sVideoPath)
    for item in range(len(li_videos)):
        ClassName=str(cnt).zfill(4)
        ClassFolderPath=FrameClassPath+"\\"+ClassName
        os.mkdir(ClassFolderPath)
        cnt+=1
    os.chdir(sPath)

def Video2Frames(sVideoPath:str)->np.array:

    oVideo = cv2.VideoCapture(sVideoPath)
    if (oVideo.isOpened() == False): raise ValueError("Error opening video file")

    liFrames = []

    # Read until video is completed
    while (True):

        (bGrabbed, arFrame) = oVideo.read()
        if bGrabbed == False: break
        # Save the resulting frame to list
        liFrames.append(arFrame)
    return np.array(liFrames)



def frames2files(arFrames:np.array, sTargetDir:str):
    """ Write array of frames to jpg files
    Input: arFrames = (number of frames, height, width, depth)
    """
    for nFrame in range(arFrames.shape[0]):
        cv2.imwrite(sTargetDir + "/frame%04d.jpg" % nFrame, arFrames[nFrame, :, :])
    return

def frames_downsample(arFrames: np.array, nFramesTarget: int) -> np.array:
    """ Adjust number of frames (eg 123) to nFramesTarget (eg 79)
    works also if originally less frames then nFramesTarget
    """

    nSamples, _, _, _ = arFrames.shape
    if nSamples == nFramesTarget: return arFrames

    # down/upsample the list of frames
    fraction = nSamples / nFramesTarget
    index = [int(fraction * i) for i in range(nFramesTarget)]
    liTarget = [arFrames[i, :, :] for i in index]
    print("Change number of frames from %d to %d" % (nSamples, nFramesTarget))
    print(index)

    return np.array(liTarget)


def ConversionVideo2Frames(sPath):
    create_folders(sPath)
    sVideosDir=sPath+"\Classes"
    sFrameDir=sPath+"\Frames"
    li_videos=os.listdir(sVideosDir)
    for item in li_videos:
        innerinput_path = sVideosDir+"\\"+item
        inner=os.listdir(innerinput_path)
        inneroutput_path = sFrameDir+"\\"+item
        for item2 in inner:
            #print(item2)
            id=item2.split(".")[0]
            FrameVideo_FolderPath=inneroutput_path+"\\"+id
            os.mkdir(FrameVideo_FolderPath)
            input_path=innerinput_path+"\\"+item2
            Frame_array=Video2Frames(input_path)
            NewFrame_array=frames_downsample(Frame_array,40)
            frames2files(NewFrame_array,FrameVideo_FolderPath)
ConversionVideo2Frames(os.getcwd())
