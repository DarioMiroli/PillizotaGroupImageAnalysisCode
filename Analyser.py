from pyzota_image_toolbox import imageTools as IT
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import stats
import matplotlib
from matplotlib import rc,rcParams
rc('axes', linewidth=2)
rc('font', weight='bold')
def getCondition(fName):
    array = fName.split(".")[0].split("_")[0:2]
    return array[0]+array[1]


#dataFolder = "./Analysis/AllData/"
dataFolder = "./Analysis/GrowthLaw"
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
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array),"Volumes":copy.deepcopy(array)}

array = [[] for i in range(numberOfFiles)]
#Each slide individually
dataDic2 = {"Names":copy.deepcopy(array),"Areas":copy.deepcopy(array),
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array),"Volumes":copy.deepcopy(array)}



for n,fName in enumerate(fileNamesPlusPath):
    f = open(fName,'r')
    dataDic["Names"].append(fName)
    dataDic2["Names"][n]=fileNames[n].split(".")[0]
    endOfFile = False
    while not endOfFile:
        try:
            d = pickle.load(f)
            mask = d["BinaryMask"]
            if mask.size > 10:
                mask = IT.ClearBorders(mask)
                area = IT.GetArea(mask)
                if area > areaCutOff:
                    length,width = IT.GetLengthAndWidth(mask,d["RawImage"])
                    volume = ((4/3.0)*np.pi*((width/2.0)**3)) + (np.pi*((width/2)**2)*(length-width))
                    condition = getCondition(fileNames[n])
                    index = conditions.index(condition)
                    dataDic["Areas"][index].append(area)
                    dataDic["Lengths"][index].append(length)
                    dataDic["Widths"][index].append(width)
                    dataDic["Volumes"][index].append(volume)

                    dataDic2["Areas"][n].append(area)
                    dataDic2["Lengths"][n].append(length)
                    dataDic2["Widths"][n].append(width)
                    dataDic2["Volumes"][n].append(volume)
        except EOFError:
            print("End of File")
            endOfFile = True

#****************************************** PLOTTING ***************************



#****************************************** PLOTTING ***************************

plt.ioff()
plt.clf()
#Plot Histograms
Props = ["Areas","Lengths","Widths","Volumes"]
#histBins = [np.linspace(0,2000,50),np.linspace(0,100,50),np.linspace(0,25,50),np.linspace(0,4000,50)]
#histArray = []
#for z,prop in enumerate(Props):
#    print(len(dataDic[prop]))
#    plt.hist([dataDic[prop][1],dataDic[prop][5]],histtype= "step",bins=histBins[z],stacked=False,label=[conditions[1],conditions[5]],alpha=0.9,normed=False)
#    plt.legend()
#    plt.title(prop)
#    plt.xlabel(prop)
#    plt.ylabel("Frequency")
#    plt.savefig("./Analysis/Graphs/"+prop+"Hist")
#    plt.show()

plt.close()

#Plot Box Plots
for prop in Props:
    data = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for n in range(len(conditions)):
        data.append(np.asarray(dataDic[prop][n])*0.051)
        ax.annotate('n={}'.format(len(data[n])), xy=(n+1, np.median(data[n])),
                xytext=(n+1, np.median(data[n])), ha='center')
    ax.boxplot(data)

    ax.set_xticklabels(conditions)
    plt.title(prop)
    ax.grid(linestyle='--', linewidth=1,axis="y")
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.savefig("./Analysis/Graphs/"+prop)
    #plt.show()

#Plot Box Plots each slide individually
for prop in Props:
    data = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for n in range(len(dataDic2["Names"])):
        data.append(np.asarray(dataDic2[prop][n])*0.05)
        ax.annotate('n={}'.format(len(data[n])), xy=(n+1, np.median(data[n])),
                xytext=(n+1, np.median(data[n])), ha='center')
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    ax.boxplot(data)
    ax.set_xticklabels(dataDic2["Names"])
    plt.title(prop)
    ax.grid(linestyle='--', linewidth=1,axis="y")
    plt.savefig("./Analysis/Graphs/All_"+prop)
    #plt.show()


plt.close("all")
plt.clf()



#******** THESIS FIGURES *************
prop = "Volumes"
meanVolumes = []
growthRates = [40,90,28]
scaleFactors = 0.051
volumes = []
yErrors = []
for n in range(len(conditions)):
    volume = np.asarray(dataDic[prop][n])*((scaleFactors)**3)
    volumes.append(volume)
    meanVolume = np.mean(volume)
    meanVolumes.append(meanVolume)
    yErr = np.std(volume)
    yErr = yErr/meanVolume
    yErrors.append((yErr/np.sqrt(len(volume))))


#************ GROWTH LAW **************
#Bottom left size vs growth rate
plt.close("all")
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(13,9))
x = np.log(2)/(np.asarray(growthRates)/60.0)
y = np.log(np.asarray(meanVolumes))
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
yPredict = x*slope+intercept

ax[1][0].errorbar(x,y,yerr=yErrors,fmt='o',markersize=10,capsize=10,elinewidth=3,ecolor="k")
ax[1][0].plot(x,yPredict,color="orange", label="y = {0:.2f}x {1:.2f} \n $S = S_0e^{{ \gamma \lambda }} = {2:.2f}e^{{ {3:.2f}\lambda }}$".format(slope,intercept,np.exp(intercept),slope),linewidth=3)
ax[1][0].plot([0,x[1]],[intercept,x[1]*slope + intercept],"--",linewidth=3,color="orange")

ax[1][0].legend(loc="upper left",fontsize="xx-large")
ax[1][0].set_xlabel("Growth rate $\mathbf{\lambda = \\frac{ln(2)}{\\tau}}$ ($h^{-1}$)",fontsize=20,fontweight="bold")
ax[1][0].set_ylabel("$\mathbf{ln}$(Mean cell volume)",fontsize=20,fontweight="bold")
ax[1][0].set_xlim(0,1.05*max(x))
ax[1][0].set_ylim(1.1*intercept,max(y)*1.5)
ax[1][0].tick_params(axis="x", labelsize=15)
ax[1][0].tick_params(axis="y", labelsize=15)
#Bottom right residuals
ax[1][1].axhline(0,color="gray",zorder=0)
ax[1][1].errorbar(x,y-yPredict,yerr=yErrors,fmt='o',markersize=10,capsize=10,elinewidth=3,ecolor="k")
ax[1][1].set_xlabel("Growth rate $\mathbf{\lambda}$ ($h^{-1}$)",fontsize=20,fontweight="bold")
ax[1][1].set_ylabel("Residuals",fontsize=20,fontweight="bold")
ax[1][1].tick_params(axis="x", labelsize=15)
ax[1][1].tick_params(axis="y", labelsize=15)
#Top left plot
for vol in volumes:
    ax[0][0].hist(vol,bins=18,alpha=0.33,density=True)
ax[0][0].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[0][0].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][0].tick_params(axis="x", labelsize=15)
ax[0][0].tick_params(axis="y", labelsize=15)
#Top right
ax[0][1].boxplot(volumes,vert=False,whis=[1,99], positions= growthRates, meanline=True )
#ax.annotate('n={}'.format(len(data[n])), xy=(n+1, np.median(data[n])),
#        xytext=(n+1, np.median(data[n])), ha='center')
ax[0][1].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[0][1].set_ylabel("Condition",fontsize=20,fontweight="bold")
ax[0][1].tick_params(axis="x", labelsize=15)
ax[0][1].tick_params(axis="y", labelsize=15)
ax[0][1].grid(linestyle='--', linewidth=1,axis="x")


fig.tight_layout()

plt.show()
