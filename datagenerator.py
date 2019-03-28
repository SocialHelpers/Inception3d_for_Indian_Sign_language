"""
This script is used for generating data for keras's model.fir_generator.
It is also used to read the class.csv file for loading the data classes.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import keras
from framesandoflow import files2frames, images_normalize


class FramesGenerator(keras.utils.Sequence):
    """
    Read and yields video frames/optical flow for Keras.model.fit_generator
    """

    def __init__(self, sPath,
                 nBatchSize, nFrames, nHeight, nWidth, nChannels,
                 liClassesFull=None, bShuffle=True):
        """
        Assume directory structure:
        ... / sPath / class / videoname / frames.jpg
        """
        self.nBatchSize = nBatchSize
        self.nFrames = nFrames
        self.nHeight = nHeight
        self.nWidth = nWidth
        self.nChannels = nChannels
        self.tuXshape = (nFrames, nHeight, nWidth, nChannels)
        self.bShuffle = bShuffle

        # retrieve all videos = frame directories
        a = []
        f = os.listdir(sPath)
        for item in f:
            inner = sPath + "/" + item
            inner_path = os.listdir(inner)
            for item2 in inner_path:
                l = inner + "/" + item2
                a.append(l)

        self.dfVideos = pd.DataFrame(sorted(a), columns=["sFrameDir"])
        self.nSamples = len(self.dfVideos)
        if self.nSamples == 0: raise ValueError("Found no frame directories files in " + sPath)
        print("Detected %d samples in %s ..." % (self.nSamples, sPath))

        # extract class labels from path
        seLabels = self.dfVideos.sFrameDir.apply(lambda s: s.split("/")[-2])
        self.dfVideos.loc[:, "sLabel"] = seLabels

        # extract list of unique classes from all detected labels
        self.liClasses = sorted(list(self.dfVideos.sLabel.unique()))

        self.nClasses = len(self.liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()  # creates instace of LabelEncoder()
        trLabelEncoder.fit(self.liClasses)  # encodes label into class value from 0 to nclasses-1
        self.dfVideos.loc[:, "nLabel"] = trLabelEncoder.transform(
            self.dfVideos.sLabel)  # convert textual labels into numerical encoded labels
        # inverse transform does the opposite

        self.on_epoch_end()
        return

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(self.nSamples / self.nBatchSize))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(self.nSamples)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, nStep):
        """
        Generate one batch of data
        """

        # Generate indexes of the batch
        indexes = self.indexes[nStep * self.nBatchSize:(nStep + 1) * self.nBatchSize]

        # get batch of videos
        dfVideosBatch = self.dfVideos.loc[indexes, :]

        nBatchSize = len(dfVideosBatch)

        # initialize arrays
        arX = np.empty((nBatchSize,) + self.tuXshape, dtype=float)
        arY = np.empty((nBatchSize), dtype=int)

        # Generate data
        for i in range(nBatchSize):
            # generate data for single video(frames)
            arX[i,], arY[i] = self.__data_generation(dfVideosBatch.iloc[i, :])

        # onehot the labels
        return arX, keras.utils.to_categorical(arY, num_classes=self.nClasses)

    def __data_generation(self, seVideo):
        """
        Returns frames for 1 video, including normalizing & preprocessing
        """

        # Get the frames from disc
        ar_nFrames = files2frames(seVideo.sFrameDir)

        # only use the first nChannels (typically 3, but maybe 2 for optical flow)
        ar_nFrames = ar_nFrames[..., 0:self.nChannels]
        ar_fFrames = images_normalize(ar_nFrames, self.nFrames, self.nHeight, self.nWidth, bRescale=True)

        return ar_fFrames, seVideo.nLabel

    def data_generation(self, seVideo):
        return self.__data_generation(seVideo)


class VideoClasses():
    """
    Loads the video classes (incl descriptions) from a csv file
    """

    def __init__(self, sClassFile: str):
        self.dfClass = pd.read_csv(sClassFile)
        self.dfClass = self.dfClass.sort_values("sClass").reset_index(drop=True)
        self.liClasses = list(self.dfClass.sClass)
        self.nClasses = len(self.dfClass)

        print("Loaded %d classes from %s" % (self.nClasses, sClassFile))
        return
