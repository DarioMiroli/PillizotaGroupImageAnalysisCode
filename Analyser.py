from pyzota_image_toolbox import imageTools as IT
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
dataFolder = "./Analysis/AllData/"
fileNames = sorted(IT.GetFileNamesFromFolder(dataFolder))
fileNamesPlusPath = sorted(IT.GetFileNamesFromFolder(dataFolder,False))
areaCutOff = 10
array = [[] for i in range(len(fileNames))]
dataDic = {"Names":copy.deepcopy(array),"Areas":copy.deepcopy(array),
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array)}
dataDic2 = {"Names":copy.deepcopy(array),"Areas":copy.deepcopy(array),
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array)}

conditions = [[],[]]
for n,fName in enumerate(fileNamesPlusPath):
    m = n
    condition = fileNames[n].split(".")[0].split("_")[0:2]
    condition = condition[0]+condition[1]
    if condition in conditions[0]:
        N = conditions[1][np.where(np.asarray(condition)==conditions[0])[0][0]]
        dataDic2["Names"][m].append(fileNames[m])
    else:
        conditions[0].append(condition)
        conditions[1].append(n)
        dataDic["Names"][n].append(fileNames[n])
        dataDic2["Names"][m].append(fileNames[m])
        N=n
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
                    dataDic["Areas"][N].append(area)
                    dataDic2["Areas"][m].append(area)
                    try:
                        length,width = IT.GetLengthAndWidth(mask)
                    except:
                        plt.show()

                    dataDic["Lengths"][N].append(length)
                    dataDic2["Lengths"][m].append(length)
                    dataDic["Widths"][N].append(width)
                    dataDic2["Widths"][m].append(width)

        except EOFError:
            print("End of File")
            endOfFile = True


plt.ioff()
plt.clf()
#Plot Histograms
Props = ["Areas","Lengths","Widths"]
histBins = [np.linspace(0,2000,50),np.linspace(0,100,50),np.linspace(0,25,50)]
for z,prop in enumerate(Props):
    for n in range(len(conditions[0])):
        plt.hist(dataDic[prop][n],bins=histBins[z],label=conditions[0][n],alpha=0.5)
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
    for n in range(len(conditions[0])):
        data.append(np.asarray(dataDic[prop][n])*0.05)
        ax.annotate('n={}'.format(len(data[n])), xy=(n+1, np.median(data[n])),
                xytext=(n+1, np.median(data[n])), ha='center')
    ax.boxplot(data)
    ax.set_xticklabels(conditions[0])
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
