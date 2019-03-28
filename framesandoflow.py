"""
This script is used for various functionalities relating to optical flow and  rgb frames.
"""
import os
import numpy as np
import warnings
import cv2


class OpticalFlow:
    def __init__(self, sAlgorithm="tvl1-fast", bThirdChannel=False, fBound=20.):
        """
        Initializes the OpticalFlow object
        :param sAlgorithm: Type of algorithm to use for calculating Optical Flow.
        :param bThirdChannel: Whether to use third channel. (Third channel is required when viewing optical flow)
        :param fBound: Upper bound for normalization.
        """
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

    def first(self, arImage):
        """
        Detects the first frame and returns initial optical flow.
        :param arImage: input image
        :return: initial optical flow.
        """
        h, w, _ = arImage.shape
        # save first image in black&white
        self.arPrev = cv2.cvtColor(arImage, cv2.COLOR_BGR2GRAY)
        # first flow = zeros
        arFlow = np.zeros((h, w, 2), dtype=np.float32)
        if self.bThirdChannel:
            self.arZeros = np.zeros((h, w, 1), dtype=np.float32)
            arFlow = np.concatenate((arFlow, self.arZeros), axis=2)
        return arFlow

    def next(self, arImage):
        """
        Calculate optical flow with respect to previous frame.
        :param arImage: input image.
        :return: calculated optical flow.
        """
        if self.arPrev.shape == (1, 1): return self.first(arImage)

        arCurrent = cv2.cvtColor(arImage, cv2.COLOR_BGR2GRAY)

        if self.sAlgorithm == "tvl1":
            arFlow = self.oTVL1.calc(self.arPrev, arCurrent, None)
        else:
            raise ValueError("Unknown optical flow type")

        arFlow = arFlow[:, :, 0:2]

        # truncate to +/-20.0, then rescale to [-1.0, 1.0]
        arFlow[arFlow > self.fBound] = self.fBound
        arFlow[arFlow < -self.fBound] = -self.fBound
        arFlow = arFlow / self.fBound

        if self.bThirdChannel:
            # add third empty channel
            arFlow = np.concatenate((arFlow, self.arZeros), axis=2)
        self.arPrev = arCurrent
        return arFlow


def frames2flows(arFrames, sAlgorithm="tvl1-fast", bThirdChannel=False, bShow=False, fBound=20.):
    """
    Converts given RGB frames to Optical Flows.
    :param arFrames: input RGB frames.
    :param sAlgorithm: Which algorithm to use for calculating Optical Flow.
    :param bThirdChannel: Whether to use third channel. (Third channel is required when viewing optical flow)
    :param bShow: Whether to display the calculated Optical flow.
    :param fBound: Upper bound for normalization.
    :return:
    """
    # initialize optical flow calculation
    oOpticalFlow = OpticalFlow(sAlgorithm=sAlgorithm, bThirdChannel=bThirdChannel, fBound=fBound)

    liFlows = []
    for i in range(len(arFrames)):
        # calc dense optical flow
        arFlow = oOpticalFlow.next(arFrames[i, ...])
        liFlows.append(arFlow)
        if bShow:
            cv2.imshow("Optical flow", flow2colorimage(arFlow))
            cv2.waitKey(1)

    return np.array(liFlows)


def flow2colorimage(ar_f_Flow):
    """
    Converts optical flow to a color image.
    :param ar_f_Flow: optical flow.
    :return: color image.
    """
    h, w, c = ar_f_Flow.shape
    if not isinstance(ar_f_Flow[0, 0, 0], np.float32):
        warnings.warn("Need to convert flows to float32")
        ar_f_Flow = ar_f_Flow.astype(np.float32)

    ar_n_hsv = np.zeros((h, w, 3), dtype=np.uint8)
    ar_n_hsv[..., 1] = 255

    # get colors
    mag, ang = cv2.cartToPolar(ar_f_Flow[..., 0], ar_f_Flow[..., 1])
    ar_n_hsv[..., 0] = ang * 180 / np.pi / 2
    ar_n_hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    ar_n_bgr = cv2.cvtColor(ar_n_hsv, cv2.COLOR_HSV2BGR)
    return ar_n_bgr


def flows2colorimages(arFlows):
    """
    Converts array of optical flow to array of color image.
    :param arFlows: optical flow.
    :return: array of color image.
    """
    n, _, _, _ = arFlows.shape
    liImages = []
    for i in range(n):
        arImage = flow2colorimage(arFlows[i, ...])
        liImages.append(arImage)
    return np.array(liImages)


def create_folders(sPath, type):
    """
    Utility to create folders for storing the Frames and Optical flow.
    :param sPath: Base path where folders are to be created
    :param type: Whether to create folders for Frames or Optical flow.
    :return: None
    """
    cnt = 1
    os.mkdir(type)
    FrameClassPath = os.path.join(sPath, type)
    os.chdir(FrameClassPath)
    sVideoPath = os.path.join(sPath, "Classes")
    li_videos = os.listdir(sVideoPath)
    for item in range(len(li_videos)):
        ClassName = str(cnt).zfill(4)
        ClassFolderPath = os.path.join(FrameClassPath, ClassName)
        print("Creating folder : ", ClassFolderPath)
        os.mkdir(ClassFolderPath)
        cnt += 1
    os.chdir(sPath)


def Video2Frames(sVideoPath):
    """
    Converts a given video into frames.
    :param sVideoPath: Path of the input video.
    :return: Frames in the given input video
    """
    oVideo = cv2.VideoCapture(sVideoPath)
    if (oVideo.isOpened() == False): raise ValueError("Error opening video file")

    liFrames = []
    while (True):
        (bGrabbed, arFrame) = oVideo.read()
        if bGrabbed == False: break
        liFrames.append(arFrame)
    return np.array(liFrames)


def files2frames(sPath):
    """
    Converts images in a given folder to frames.
    :param sPath: Path of folder containing the images.
    :return: Frames of images present in the given input folder.
    """
    a = os.listdir(sPath)
    b = [item for item in a if item.endswith(".jpg")]
    liFiles = sorted(b)
    if len(liFiles) == 0: raise ValueError("No frames found in " + str(sPath))

    liFrames = []
    for sFramePath in liFiles:
        a = sPath + "/" + sFramePath
        arFrame = cv2.imread(a)
        liFrames.append(arFrame)
        # print(liFrames)

    return np.array(liFrames)


def image_crop(arFrame, nHeightTarget, nWidthTarget):
    """
    Crop each frame in array to specified size.
    :param arFrames: Array of frames.
    :param nHeightTarget: Target Height.
    :param nWidthTarget: Target Width.
    :return: Array of cropped frames.
    """
    nHeight, nWidth, nDepth = arFrame.shape

    sX = int(nWidth / 2 - nWidthTarget / 2)
    sY = int(nHeight / 2 - nHeightTarget / 2)

    arFrames = arFrame[sY:sY + nHeightTarget, sX:sX + nWidthTarget, :]

    return arFrames


def frames_show(arFrames: np.array, nWaitMilliSec: int = 100):
    """
    Displays OpenCV frames.
    :param arFrames: array of frames to display.
    :param nWaitMilliSec: duration to display the frames.
    :return:
    """
    nFrames, nHeight, nWidth, nDepth = arFrames.shape

    for i in range(nFrames):
        cv2.imshow("Frame", arFrames[i, :, :, :])
        cv2.waitKey(nWaitMilliSec)

    return


def images_crop(arFrames, nHeightTarget, nWidthTarget):
    """
    Crop each frame in array to specified size.
    :param arFrames: Array of frames.
    :param nHeightTarget: Target Height.
    :param nWidthTarget: Target Width.
    :return: Array of cropped frames.
    """
    nSamples, nHeight, nWidth, nDepth = arFrames.shape

    sX = int(nWidth / 2 - nWidthTarget / 2)
    sY = int(nHeight / 2 - nHeightTarget / 2)

    arFrames = arFrames[:, sY:sY + nHeightTarget, sX:sX + nWidthTarget, :]

    return arFrames


def images_rescale(arFrames):
    """
    Rescale array of images (rgb 0-255) to [-1.0, 1.0]
    :param arFrames: Array of frames.
    :return: Array of rescaled frames.
    """

    ar_fFrames = arFrames / 127.5
    ar_fFrames -= 1.

    return ar_fFrames


def images_normalize(arFrames, nFrames, nHeight, nWidth, bRescale=True):
    """
    Normalizes images using:
        - downsample/upsample number of frames
        - crop to centered image
        - rescale rgb 0-255 value to [-1.0, 1.0]
    :param arFrames: array of input images.
    :param nFrames: downsample/upsample to target number of frames.
    :param nHeight: target height of images.
    :param nWidth: target width of images.
    :param bRescale: whether to rescale the images or not.
    :return: array of normalized images.
    """
    # normalize the number of frames (assuming typically downsampling)
    arFrames = frames_downsample(arFrames, nFrames)

    # crop to centered image
    arFrames = images_crop(arFrames, nHeight, nWidth)

    if bRescale:
        # normalize to [-1.0, 1.0]
        arFrames = images_rescale(arFrames)
    else:
        if np.max(np.abs(arFrames)) > 1.0: warnings.warn("Images not normalized")

    return arFrames


def frames2files(arFrames, sTargetDir):
    """
    Write array of frames to jpg files.
    :param arFrames: array of input images
    :param sTargetDir: path of directory where image files are to be stored.
    :return: None
    """
    for nFrame in range(arFrames.shape[0]):
        cv2.imwrite(sTargetDir + "/frame%04d.jpg" % nFrame, arFrames[nFrame, :, :])
    return


def flows2file(arFlows, sTargetDir):
    """
    Saves array of flows to jpg files.
    :param arFlows: array of Optical flow.
    :param sTargetDir: Path of directory where Optical flow image files are to be stored.
    :return: None
    """
    n, h, w, c = arFlows.shape
    # os.makedirs(sTargetDir, exist_ok=True)
    arZeros = np.zeros((h, w, 1), dtype=np.float32)
    for i in range(n):
        # add third empty channel
        ar_f_Flow = np.concatenate((arFlows[i, ...], arZeros), axis=2)
        ar_n_Flow = np.round((ar_f_Flow + 1.0) * 127.5).astype(np.uint8)
        cv2.imwrite(sTargetDir + "/flow%03d.jpg" % (i), ar_n_Flow)
    return


def frames_downsample(arFrames, nFramesTarget):
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
    """
    Convert videos into RGB frames and Optical flow and store them on disk.
    :param sPath: Path of directory where videos are stored.
    :return: None
    """
    create_folders(sPath, "Frames")
    create_folders(sPath, "OFlows")
    sVideosDir = os.path.join(sPath, "Classes")
    sFrameDir = os.path.join(sPath, "Frames")
    sOflowDir = os.path.join(sPath, "OFlows")
    li_videos = os.listdir(sVideosDir)
    for class_id in li_videos:
        class_path = os.path.join(sVideosDir, class_id)
        class_videos = os.listdir(class_path)
        class_frame_output_path = os.path.join(sFrameDir, class_id)
        class_flow_output_path = os.path.join(sOflowDir, class_id)
        for video in class_videos:
            id = video.split(".")[0]
            frame_output_path = os.path.join(class_frame_output_path, id)
            flow_output_path = os.path.join(class_flow_output_path, id)
            os.mkdir(frame_output_path)
            os.mkdir(flow_output_path)
            input_video = os.path.join(class_path, video)
            print("Converting Video : ", input_video)
            print("Saving Frames to : ", frame_output_path)
            print("Saving OFlows to : ", flow_output_path)
            frame_array = Video2Frames(input_video)
            downsampled_frame_array = frames_downsample(frame_array, 40)
            oflows = frames2flows(downsampled_frame_array, sAlgorithm="tvl1-fast", bThirdChannel=False, bShow=False,
                                  fBound=20.)
            frames2files(downsampled_frame_array, frame_output_path)
            flows2file(oflows, flow_output_path)


if "_name_" == "_main_":
    ConversionVideo(os.getcwd())
