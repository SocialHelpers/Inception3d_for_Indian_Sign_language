import os
import shutil
import numpy as np
import pandas as pd
import warnings
from imutils import paths
import cv2

def create_folders(sPath):
    li_videos=[]
    cnt = 1
    os.mkdir("Classes")
    ClassPath = sPath + "\Classes"
    os.chdir(ClassPath)
    sVideoPath=sPath+"\Augmented_FinalDataset"
    li_videos=os.listdir(sVideoPath)
    for item in range(len(li_videos)*5):
        ClassName=str(cnt).zfill(4)
        ClassFolderPath=ClassPath+"\\"+ClassName
        os.mkdir(ClassFolderPath)
        cnt+=1
    os.chdir(sPath)
def folders2classes(sFolderPath):
    print(sFolderPath)
    sClassPath = sFolderPath + "\\"+"Classes"
    os.chdir(sClassPath)
    pi_videos=os.listdir(sClassPath)
    os.chdir(sFolderPath)
    li_videos=paths.list_files("Augmented_FinalDataset")
    cnt=0
    for item in li_videos:
        cnt += 1
        id = item.split(".")[0]
        pid=id[-6:-2]
        #print(item)
        #print(pid)
        if pid in pi_videos:
            sVideoName = str(cnt).zfill(4)
            sClassSavePath=sClassPath+"\\"+pid+"\\"+sVideoName+".mp4"
            #print(sClassSavePath)
            shutil.copy(src=item,dst=sClassSavePath)

str1 = os.getcwd()
create_folders(str1)
folders2classes(str1)
