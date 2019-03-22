"""
This script provides functionality to resize videos to (320,240).
It also reduces the frame rate to 10 fps.
It uses "ffmpeg" for the above stated purpose.
"""

import os

print("\n Resizing and Analyzing fps from %s ..." % (sVideoDir))

sVideoDir = os.path.join(os.getcwd(), "Dataset")
print(sVideoDir)
liVideos = []
pathlist = os.listdir(sVideoDir)
Folders = []

for item in pathlist:
    inner = os.path.join(sVideoDir, item)
    innerpaths = os.listdir(inner)
    Folders.append(inner)
    for item2 in innerpaths:
        inner2 = os.path.join(inner, item2)
        liVideos.append(inner2)

cmd = "ffmpeg -y -i {} -r 25 -s 320x240 -r 10 {}"

os.mkdir("FinalDataset")
cwd = os.getcwd()
os.chdir("FinalDataset")

f = sorted(set(Folders))
folders = []

for item in f:
    a = item.split("\\")
    os.mkdir(a[-1])
    ce = os.getcwd()
    ce1 = os.path.join(ce, a[-1])
    folders.append(ce1)

os.chdir(cwd)
j = 0
cnt = 1

for item in liVideos:
    source = item
    id = str(cnt).zfill(4) + ".mp4"
    dest = folders[j]
    dest = os.path.join(dest, id)
    formatted_cmd = cmd.format(source, dest)
    print(formatted_cmd)
    os.system(formatted_cmd)
    cnt += 1
    if cnt == 106:
        cnt = 1
        j += 1
