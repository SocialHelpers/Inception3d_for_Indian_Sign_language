import os
import shutil
path=os.path.join(os.getcwd(),"Dataset\Rahul")

paths=os.listdir(path)

#os.mkdir("Rahul1")
for item in paths:
	source=os.path.join(path,item)
	id=item.split(".")[0]
	destid="DSC_"+id.zfill(4)
	dest=os.path.join(os.getcwd(),"Dataset\Rahul1")
	dest=os.path.join(dest,destid)+".MOV"
	print(source)
	print(dest)
	shutil.copy(src=source,dst=dest)