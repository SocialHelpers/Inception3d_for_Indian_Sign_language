import os
import shutil
from imutils import paths

def create_folders(sPath):
    cnt = 1
    os.mkdir("Classes")
    ClassPath = os.path.join(sPath,"Classes")
    os.chdir(ClassPath)
    for item in range(105):
        ClassName=str(cnt).zfill(4)
        ClassFolderPath=os.path.join(ClassPath,ClassName)
        print("Creating folder : ",ClassFolderPath)
        os.mkdir(ClassFolderPath)
        cnt+=1
    os.chdir(sPath)

def folders2classes(sFolderPath):
    sClassPath = os.path.join(sFolderPath,"Classes")
    pi_videos=os.listdir(sClassPath)
    li_videos=paths.list_files("Augmented_FinalDataset")
    cnt=0
    for item in li_videos:
        cnt += 1
        id = item.split(".")[0]
        class_id=id[-6:-2]
        if class_id in pi_videos:
            sVideoName = str(cnt).zfill(4)+".mp4"
            sClassSavePath=os.path.join(sClassPath,os.path.join(class_id,sVideoName))
            print("Adding video to : ",sClassSavePath)
            shutil.copy(src=item,dst=sClassSavePath)

cwd = os.getcwd()
create_folders(cwd)
folders2classes(cwd)
