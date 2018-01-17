from pyzota_image_toolbox import imageTools as IT
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy

def getCondition(fName):
    array = fName.split(".")[0].split("_")[0:2]
    return array[0]+array[1]


dataFolder = "./Analysis/AllData/"
fileNames = sorted(IT.GetFileNamesFromFolder(dataFolder))
fileNamesPlusPath = sorted(IT.GetFileNamesFromFolder(dataFolder,False))
areaCutOff = 10


numberOfFiles = len(fileNames)
print("Number of files = {}".format(numberOfFiles))
conditions =sorted(list(set([ getCondition(i) for i in fileNames ])))
print("Conditions={}".format(conditions))


array = [[] for i in range(len(conditions))]
#Conditions grouped together
dataDic = {"Names":copy.deepcopy(array),"Areas":copy.deepcopy(array),
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array)}

array = [[] for i in range(numberOfFiles)]
#Each slide individually
dataDic2 = {"Names":copy.deepcopy(array),"Areas":copy.deepcopy(array),
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array)}



for n,fName in enumerate(fileNamesPlusPath):
    f = open(fName,'r')
    endOfFile = False
    while not endOfFile:
        try:
            d = pickle.load(f)
            mask = d["BinaryMask"]
            if mask.size > 10:
                mask = IT.ClearBorders(mask)
                area = IT.GetArea(mask)
                if area > areaCutOff:
                    length,width = IT.GetLengthAndWidth(mask)
                    condition = getCondition(fileNames[n])
                    index = conditions.index(condition)
                    dataDic["Areas"][index].append(area)
                    dataDic["Lengths"][index].append(length)
                    dataDic["Widths"][index].append(width)
                    dataDic2["Areas"][n].append(area)
                    dataDic2["Lengths"][n].append(length)
                    dataDic2["Widths"][n].append(width)
        except EOFError:
            print("End of File")
            endOfFile = True

#****************************************** PLOTTING ***************************



#****************************************** PLOTTING ***************************


plt.ioff()
plt.clf()
#Plot Histograms
Props = ["Areas","Lengths","Widths"]
histBins = [np.linspace(0,2000,50),np.linspace(0,100,50),np.linspace(0,25,50)]
for z,prop in enumerate(Props):
    for n in range(len(conditions)):
        plt.hist(dataDic[prop][n],bins=histBins[z],label=conditions[n],alpha=0.5)
    plt.legend()
    plt.title(prop)
    plt.xlabel(prop)
    plt.ylabel("Frequency")
    plt.savefig("./Analysis/Graphs/"+prop+"Hist")
    plt.show()

#Plot Box Plots
for prop in Props:
    data = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for n in range(len(conditions)):
        data.append(np.asarray(dataDic[prop][n])*0.05)
        ax.annotate('n={}'.format(len(data[n])), xy=(n+1, np.median(data[n])),
                xytext=(n+1, np.median(data[n])), ha='center')
    ax.boxplot(data)
    ax.set_xticklabels(conditions)
    plt.title(prop)
    ax.grid(linestyle='--', linewidth=1,axis="y")
    plt.savefig("./Analysis/Graphs/"+prop)
    plt.show()

#Plot Box Plots each slide individually
for prop in Props:
    data = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for n in range(len(dataDic2["Names"])):
        data.append(np.asarray(dataDic2[prop][n])*0.05)
        ax.annotate('n={}'.format(len(data[n])), xy=(n+1, np.median(data[n])),
                xytext=(n+1, np.median(data[n])), ha='center')
    ax.boxplot(data)
    ax.set_xticklabels(dataDic2["Names"])
    plt.title(prop)
    ax.grid(linestyle='--', linewidth=1,axis="y")
    plt.savefig("./Analysis/Graphs/All_"+prop)
    plt.show()
