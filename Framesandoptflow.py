import os
import shutil
import numpy as np
import pandas as pd
import warnings
from imutils import paths
import cv2

class OpticalFlow:
    """ Initialize an OpticalFlow object,
    then use next() to calculate optical flow from subsequent frames.
    Detects first call automatically.
    """
    def __init__(self, sAlgorithm: str = "tvl1-fast", bThirdChannel: bool = False, fBound: float = 20.):
        self.bThirdChannel = bThirdChannel
        self.fBound = fBound
        self.arPrev = np.zeros((1, 1))

        if sAlgorithm == "tvl1-fast":
            self.oTVL1 = cv2.DualTVL1OpticalFlow_create(scaleStep=0.5, warps=3, epsilon=0.02)
            self.sAlgorithm = "tvl1"

        elif sAlgorithm == "tvl1-quality":
            self.oTVL1 = cv2.DualTVL1OpticalFlow_create()
            # Default: (tau=0.25, lambda=0.15, theta=0.3, nscales=5, warps=5, epsilon=0.01, scaleStep=0.5)
            self.sAlgorithm = "tvl1"
        return

    def first(self, arImage: np.array) -> np.array:
        h, w, _ = arImage.shape
        # save first image in black&white
        self.arPrev = cv2.cvtColor(arImage, cv2.COLOR_BGR2GRAY)
        # first flow = zeros
        arFlow = np.zeros((h, w, 2), dtype=np.float32)
        if self.bThirdChannel:
            self.arZeros = np.zeros((h, w, 1), dtype=np.float32)
            arFlow = np.concatenate((arFlow, self.arZeros), axis=2)
        return arFlow

    def next(self, arImage: np.array) -> np.array:
        # first?
        if self.arPrev.shape == (1, 1): return self.first(arImage)
        # get image in black&white
        arCurrent = cv2.cvtColor(arImage, cv2.COLOR_BGR2GRAY)
        if self.sAlgorithm == "tvl1":
            arFlow = self.oTVL1.calc(self.arPrev, arCurrent, None)
        else:
            raise ValueError("Unknown optical flow type")

        # only 2 dims
        arFlow = arFlow[:, :, 0:2]

        # truncate to +/-15.0, then rescale to [-1.0, 1.0]
        arFlow[arFlow > self.fBound] = self.fBound
        arFlow[arFlow < -self.fBound] = -self.fBound
        arFlow = arFlow / self.fBound

        if self.bThirdChannel:
            # add third empty channel
            arFlow = np.concatenate((arFlow, self.arZeros), axis=2)
        self.arPrev = arCurrent
        return arFlow


def frames2flows(arFrames: np.array(int), sAlgorithm="tvl1-fast", bThirdChannel: bool = False, bShow=False,
                 fBound: float = 20.) -> np.array(float):
    """ Calculates optical flow from frames

    Returns:
        array of flow-arrays, each with dim (h, w, 2),
        with "flow"-values truncated to [-15.0, 15.0] and then scaled to [-1.0, 1.0]
        If bThirdChannel = True a third channel with zeros is added
    """

    # initialize optical flow calculation
    oOpticalFlow = OpticalFlow(sAlgorithm=sAlgorithm, bThirdChannel=bThirdChannel, fBound=fBound)

    liFlows = []
    # loop through all frames
    for i in range(len(arFrames)):
        # calc dense optical flow
        arFlow = oOpticalFlow.next(arFrames[i, ...])
        liFlows.append(arFlow)
        if bShow:
            cv2.imshow("Optical flow", flow2colorimage(arFlow))
            cv2.waitKey(1)

    return np.array(liFlows)

def create_folders(sPath,type):
    cnt = 1
    os.mkdir(type)
    FrameClassPath = os.path.join(sPath,type)
    os.chdir(FrameClassPath)
    sVideoPath=os.path.join(sPath,"Classes")
    li_videos=os.listdir(sVideoPath)
    for item in range(len(li_videos)):
        ClassName=str(cnt).zfill(4)
        ClassFolderPath=os.path.join(FrameClassPath,ClassName)
        print("Creating folder : ",ClassFolderPath)
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


def flows2file(arFlows:np.array(float), sTargetDir:str):
    """ Save array of flows (2 channels with values in [-1.0, 1.0])
    to jpg files (with 3 channels 0-255 each) in sTargetDir
    """
    n, h, w, c = arFlows.shape
    #os.makedirs(sTargetDir, exist_ok=True)
    arZeros = np.zeros((h, w, 1), dtype = np.float32)
    for i in range(n):
        # add third empty channel
        ar_f_Flow = np.concatenate((arFlows[i, ...], arZeros), axis=2)
        # rescale to 0-255
        ar_n_Flow = np.round((ar_f_Flow + 1.0) * 127.5).astype(np.uint8)
        cv2.imwrite(sTargetDir + "/flow%03d.jpg"%(i), ar_n_Flow)
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

    return np.array(liTarget)


def ConversionVideo(sPath):
    create_folders(sPath,"Frames")
    create_folders(sPath, "OFlows")
    sVideosDir=os.path.join(sPath,"Classes")
    sFrameDir=os.path.join(sPath,"Frames")
    sOflowDir=os.path.join(sPath,"OFlows")
    li_videos=os.listdir(sVideosDir)
    for class_id in li_videos:
        class_path = os.path.join(sVideosDir,class_id)
        class_videos=os.listdir(class_path)
        class_frame_output_path = os.path.join(sFrameDir,class_id)
        class_flow_output_path = os.path.join(sOflowDir, class_id)
        for video in class_videos:
            id=video.split(".")[0]
            frame_output_path=os.path.join(class_frame_output_path,id)
            flow_output_path = os.path.join(class_flow_output_path, id)
            os.mkdir(frame_output_path)
            os.mkdir(flow_output_path)
            input_video=os.path.join(class_path,video)
            print("Converting Video : ", input_video)
            print("Saving Frames to : ",frame_output_path)
            print("Saving OFlows to : ", flow_output_path)
            frame_array=Video2Frames(input_video)
            downsampled_frame_array=frames_downsample(frame_array,40)
            oflows = frames2flows(downsampled_frame_array, sAlgorithm="tvl1-fast", bThirdChannel=False, bShow=False, fBound=20.)
            frames2files(downsampled_frame_array,frame_output_path)
            flows2file(oflows, flow_output_path)

ConversionVideo(os.getcwd())
