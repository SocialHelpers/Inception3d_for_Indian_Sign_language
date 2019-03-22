"""
This script is used for splitting the entire data-set into training and validation parts.
"""
import os
from imutils import paths
from random import shuffle

homedir = os.getcwd()

oflowdir = os.path.join(homedir, "OFlows")

class_list = os.listdir(oflowdir)

finaldir = os.path.join(homedir, "Training_data")
os.mkdir(finaldir)
train_dir = os.path.join(finaldir, "train")
val_dir = os.path.join(finaldir, "val")
os.mkdir(train_dir)
os.mkdir(val_dir)
for class_id in class_list:
    class_dir = os.path.join(oflowdir, class_id)
    videos = os.listdir(class_dir)
    shuffle(videos)
    train = videos[0:int(len(videos) * 0.8)]
    val = list(set(videos) - set(train))
    os.mkdir(os.path.join(train_dir, class_id))
    os.mkdir(os.path.join(val_dir, class_id))
    for train_video in train:
        source = os.path.join(class_dir, train_video)
        destination = os.path.join(train_dir, os.path.join(class_id, train_video))
        print(source)
        print(destination)
        os.rename(src=source, dst=destination)
    for val_video in val:
        source = os.path.join(class_dir, val_video)
        destination = os.path.join(val_dir, os.path.join(class_id, val_video))
        print(source)
        print(destination)
        os.rename(src=source, dst=destination)
