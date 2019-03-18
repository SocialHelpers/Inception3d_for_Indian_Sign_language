from vidaug import augmentors as va
import numpy as np
import cv2
from imutils import paths
import os

def video2array(path):
    cap=cv2.VideoCapture(path)

    data=[]
    while(cap.isOpened()):
        ret,frame=cap.read()
        if ret:
            data.append(frame)
        else:
            break

    data=np.array(data)
    return data

operations=[None,va.Sequential([va.RandomRotate(degrees=30)]),va.Add(100),
            va.PiecewiseAffineTransform(displacement_magnification=9,displacement_kernel=45,displacement=8),
           va.Sequential([va.RandomTranslate(x=40,y=20)]),va.Sequential([va.Multiply(0.5)])]

list=paths.list_files("FinalDataset")

#os.mkdir("Augmented_FinalDataset")
for item in list:
    #print(item)
    id=item.split(".")[0]
    array=video2array(item)
    for i in range(len(operations)):
        path = "Augmented_"+id + "_" + str(i + 1) + ".mp4"
        print(path)
        #os.makedirs(path[:-11],exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(path, fourcc, 10.0, (320, 240))
        if operations[i]!=None:
            modifier = operations[i]
            modified_output=modifier(array)
        else:
            modified_output=array
        for j in range(len(modified_output)):
            #cv2.imshow("im", modified_output[j])
            out.write(modified_output[j])
            #cv2.waitKey(1)
        out.release()
