from vidaug import augmentors as va
import numpy as np
import cv2

cap=cv2.VideoCapture("0001.mp4")

data=[]
while(cap.isOpened()):
    ret,frame=cap.read()
    if ret:
        data.append(frame)
    else:
        break

data=np.array(data)

#seq=va.RandomRotate(degrees=180)
seq=va.Sequential([va.InvertColor()])

im=seq(data)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 10.0, (320,240))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output2.mp4',fourcc, 10.0, (320,240))
for i in range(len(im)):
    cv2.imshow("im",im[i])
    out.write(im[i])
    cv2.waitKey(100)

out.release()