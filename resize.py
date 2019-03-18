import os

#print("\nAnalyze video durations and fps from %s ..." % (sVideoDir))
sVideoDir=os.path.join(os.getcwd(),"Dataset")
print(sVideoDir)
liVideos = []
pathlist=os.listdir(sVideoDir)
Folders=[]
for item in pathlist:
	inner = os.path.join(sVideoDir,item)
	innerpaths=os.listdir(inner)
	Folders.append(inner)
	for item2 in innerpaths:
		inner2=os.path.join(inner,item2)
		liVideos.append(inner2)

cmd="ffmpeg -y -i {} -r 25 -s 320x240 -r 10 {}"

os.mkdir("FinalDataset")
cwd=os.getcwd()
os.chdir("FinalDataset")

f=sorted(set(Folders))
folders=[]
for item in f:
	a=item.split("\\")
	#print(a[-1])
	os.mkdir(a[-1])
	ce=os.getcwd()
	ce1=os.path.join(ce,a[-1])
	folders.append(ce1)
os.chdir(cwd)
if len(liVideos) == 0: raise ValueError("No videos detected")
j=0
cnt=1
for item in liVideos:
	source=item
	id = str(cnt).zfill(4) + ".mp4"
	dest=folders[j]
	dest=os.path.join(dest,id)
	cmd1 = cmd.format(source,dest)
	print(cmd1)
	os.system(cmd1)
	cnt+=1
	if cnt==106:
		cnt=1
		j+=1
