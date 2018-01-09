from pyzota_image_toolbox import imageTools as IT
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
dataFolder = "./Analysis/AllData/"
fileNames = IT.GetFileNamesFromFolder(dataFolder)
fileNamesPlusPath = IT.GetFileNamesFromFolder(dataFolder,False)
areaCutOff = 10
array = [[] for i in range(len(fileNames))]
print(array)
dataDic = {"Names":copy.deepcopy(array),"Areas":copy.deepcopy(array),
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array)}
for n,fName in enumerate(fileNamesPlusPath):
    dataDic["Names"][n].append(fileNames[n])
    print(dataDic)
    f = open(fName,'r')
    endOfFile = False
    jj=0
    while not endOfFile:
        try:
            d = pickle.load(f)
            mask = d["BinaryMask"]
            if mask.size > 10:
                mask = IT.ClearBorders(mask)
                area = IT.GetArea(mask)
                jj+=1
                if area > areaCutOff:
                    dataDic["Areas"][n].append(area)
                    length,width = IT.GetLengthAndWidth(mask,jj)
                    dataDic["Lengths"][n].append(length)
                    dataDic["Widths"][n].append(width)

        except EOFError:
            print("End of File")
            endOfFile = True

#Plot Histograms
plt.ioff()
plt.clf()

for n in range(len(dataDic["Names"])):
    plt.hist(dataDic["Areas"][n],bins=np.linspace(0,2000,50),label=dataDic["Names"][n])
plt.legend()
plt.show()

for n in range(len(dataDic["Names"])):
    plt.hist(dataDic["Lengths"][n],bins=np.linspace(0,100,50),label=dataDic["Names"][n])
plt.legend()
plt.show()

for n in range(len(dataDic["Names"])):
    plt.hist(dataDic["Widths"][n],bins=np.linspace(0,25,50),label=dataDic["Names"][n])
plt.legend()
plt.show()
