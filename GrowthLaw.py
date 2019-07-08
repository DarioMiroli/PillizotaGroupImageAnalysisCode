from pyzota_image_toolbox import imageTools as IT
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import stats
import matplotlib
from matplotlib import rc,rcParams
from matplotlib.patches import Rectangle
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
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array),"Volumes":copy.deepcopy(array), "Intensities":copy.deepcopy(array)}

array = [[] for i in range(numberOfFiles)]
#Each slide individually
dataDic2 = {"Names":copy.deepcopy(array),"Areas":copy.deepcopy(array),
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array),"Volumes":copy.deepcopy(array), "Intensities":copy.deepcopy(array)}



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
                    meanIntensity = IT.getMeanIntensity(mask,d["RawImage"])
                    volume = ((4/3.0)*np.pi*((width/2.0)**3)) + (np.pi*((width/2)**2)*(length-width))
                    condition = getCondition(fileNames[n])
                    index = conditions.index(condition)
                    dataDic["Areas"][index].append(area)
                    dataDic["Lengths"][index].append(length)
                    dataDic["Widths"][index].append(width)
                    dataDic["Volumes"][index].append(volume)
                    dataDic["Intensities"][index].append(meanIntensity)

                    dataDic2["Areas"][n].append(area)
                    dataDic2["Lengths"][n].append(length)
                    dataDic2["Widths"][n].append(width)
                    dataDic2["Volumes"][n].append(volume)
                    dataDic2["Intensities"][index].append(meanIntensity)

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

from matplotlib.patches import Rectangle

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
colors = ['C0',"C1","C2"]
#Bottom left size vs growth rate
plt.close("all")
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(13,9))
ax[0][0].text(0.82, 0.85 , "A", transform=ax[0][0].transAxes, size=30, weight='bold')
ax[0][1].text(0.82, 0.85 , "B", transform=ax[0][1].transAxes, size=30, weight='bold')
ax[1][0].text(0.05, 0.85 , "C", transform=ax[1][0].transAxes, size=30, weight='bold')
ax[1][1].text(0.82, 0.85 , "D", transform=ax[1][1].transAxes, size=30, weight='bold')
x = np.log(2)/(np.asarray(growthRates)/60.0)
y = np.log(np.asarray(meanVolumes))
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
yPredict = x*slope+intercept

ax[1][0].errorbar(x,y,yerr=yErrors,fmt="none",markersize=10,capsize=10,elinewidth=3,ecolor="k")
print(x)
print(y)
print(yErrors)
ax[1][0].scatter(x,y,s=90,c=colors,zorder=99,alpha=1.0,marker="D")
ax[1][0].plot(x,yPredict,color="C9", label="y = {0:.2f}x {1:.2f} \n $S = S_0e^{{ \gamma \lambda }} = {2:.2f}e^{{ {3:.2f}\lambda }}$".format(slope,intercept,np.exp(intercept),slope),linewidth=3)
ax[1][0].plot([0,x[1]],[intercept,x[1]*slope + intercept],"--",linewidth=3,color="C9")

ax[1][0].legend(loc="lower right",fontsize="xx-large")
ax[1][0].set_xlabel("Growth rate $\mathbf{\lambda = \\frac{ln(2)}{\\tau}}$ ($h^{-1}$)",fontsize=20,fontweight="bold")
ax[1][0].set_ylabel("$\mathbf{ln}$(Mean cell volume)",fontsize=20,fontweight="bold")
ax[1][0].set_xlim(0,1.05*max(x))
ax[1][0].set_ylim(1.1*intercept,max(y)*1.5)
ax[1][0].tick_params(axis="x", labelsize=15)
ax[1][0].tick_params(axis="y", labelsize=15)
#Bottom right residuals
ax[1][1].axhline(0,color="gray",zorder=0)
ax[1][1].errorbar(x,y-yPredict,yerr=yErrors,fmt='none',markersize=10,capsize=10,elinewidth=3,ecolor="k")
ax[1][1].scatter(x,y-yPredict,s=90,c=colors,zorder=99,alpha=1.0,marker="D")
ax[1][1].set_xlabel("Growth rate $\mathbf{\lambda}$ ($h^{-1}$)",fontsize=20,fontweight="bold")
ax[1][1].set_ylabel("Residuals",fontsize=20,fontweight="bold")
ax[1][1].tick_params(axis="x", labelsize=15)
ax[1][1].tick_params(axis="y", labelsize=15)
#Top left plot
labels = ["M63 + Glu + CAA, N = {}".format(len(volumes[0])),"M63 + Glycerol, N = {}".format(len(volumes[1])),"RDM, N = {}".format(len(volumes[2])) ]
for i,vol in enumerate(volumes):
    ax[0][0].hist(vol,bins=18,alpha=0.33,density=True,label=labels[i])
ax[0][0].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[0][0].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][0].tick_params(axis="x", labelsize=15)
ax[0][0].tick_params(axis="y", labelsize=15)
#ax[0][0].legend(fontsize="x-large")
#Top right

box = ax[0][1].boxplot(volumes,vert=False,whis=[1,99], positions= growthRates,widths=10 , patch_artist=True )
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.33)

ax[0][1].set_ylim(0,110)
ax[0][1].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[0][1].set_ylabel("Doubling time $\\tau$ (min)",fontsize=20,fontweight="bold")
ax[0][1].set_yticks([20,40,60,80,100])
ax[0][1].set_yticklabels([20,40,60,80,100])
legendItems = []
for color in colors:
    legendItems.append(Rectangle((0, 0), 1, 1, fc=color,alpha=0.33, fill=True, edgecolor='none', linewidth=0))
for i in range(len(volumes)):
    ax[0][1].plot(np.mean(volumes[i]),growthRates[i],'D',alpha=0.5,color=colors[i],markersize=7)
#ax[0][1].legend(legendItems, labels,fontsize="large")
#fig.legend(legendItems, labels,fontsize="xx-large",ncol=3,loc="upper center")#loc = (0.05, 0.97))
fig.legend([legendItems[1],legendItems[0],legendItems[2]], [labels[1],labels[0],labels[2]],loc='upper center', bbox_to_anchor=(0.5, 1.03), fancybox=False, shadow=False, ncol=3,fontsize="xx-large")
ax[0][1].tick_params(axis="x", labelsize=15)
ax[0][1].tick_params(axis="y", labelsize=15)
ax[0][1].grid(linestyle='--', linewidth=1,axis="x")
fig.tight_layout()
fig.savefig('ThesisGraphs/GrowthLaw/GrowthLaw.pdf', bbox_inches='tight')
fig.savefig('ThesisGraphs/GrowthLaw/GrowthLaw.png', bbox_inches='tight')
#plt.show()
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************



#**************** WIDTH AND LENGTH COMAPRISONS ******************************#
lengths = []
meanLengths  = []
widths = []
meanWidths = []
surfaceAreas = []
meanSurfaceAreas = []
intensities = []
meanIntensities = []
scaleFactors = 0.051
for n in range(len(conditions)):
    length = np.asarray(dataDic["Lengths"][n])*((scaleFactors))
    lengths.append(length)
    meanLengths.append(np.mean(length))
    width = np.asarray(dataDic["Widths"][n])*((scaleFactors))
    widths.append(width)
    surfaceArea = ((width**2)*np.pi)  + ((length-width)*width)
    surfaceAreas.append(surfaceArea)
    meanSurfaceAreas.append(np.mean(surfaceArea))
    meanWidths.append(np.mean(width))
    intensity = np.asarray(dataDic["Intensities"][n])
    intensities.append(intensity)
    meanIntensities.append(np.mean(intensity))

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(13,9))
ax[0][0].text(0.05, 0.85 , "A", transform=ax[0][0].transAxes, size=30, weight='bold')
ax[0][1].text(0.05, 0.85 , "B", transform=ax[0][1].transAxes, size=30, weight='bold')
ax[1][0].text(0.05, 0.85 , "C", transform=ax[1][0].transAxes, size=30, weight='bold')
ax[1][1].text(0.05, 0.85 , "D", transform=ax[1][1].transAxes, size=30, weight='bold')


box = ax[1][0].boxplot(lengths,vert=False,whis=[1,99], positions= growthRates,widths=10 , patch_artist=True )
ax[1][0].grid(linestyle='--', linewidth=1,axis="x")
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.33)
for i in range(len(lengths)):
    ax[1][0].plot(np.mean(lengths[i]),growthRates[i],'o',alpha=0.5,color=colors[i],markersize=7)
ax[1][0].set_ylim(20,100)
ax[1][0].set_xlim(0,7)
ax[1][0].set_xlabel("Length($\\mu m$)",fontsize=20,fontweight="bold")
ax[1][0].set_ylabel("Doubling time $\\tau$ (min)",fontsize=20,fontweight="bold")
ax[1][0].set_yticks([20,40,60,80,100])
ax[1][0].set_yticklabels([20,40,60,80,100])
ax[1][0].tick_params(axis="x", labelsize=15)
ax[1][0].tick_params(axis="y", labelsize=15)


box = ax[1][1].boxplot(widths,vert=False,whis=[1,99], positions= growthRates,widths=10 , patch_artist=True )
ax[1][1].grid(linestyle='--', linewidth=1,axis="x")

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.33)
for i in range(len(widths)):
    ax[1][1].plot(np.mean(widths[i]),growthRates[i],'o',alpha=0.5,color=colors[i],markersize=7)
ax[1][1].set_ylim(20,100)
ax[1][1].set_xlabel("Width ($\\mu m$)",fontsize=20,fontweight="bold")
ax[1][1].set_ylabel("Doubling time $\\tau$ (min)",fontsize=20,fontweight="bold")
ax[1][1].set_yticks([20,40,60,80,100])
ax[1][1].set_yticklabels([20,40,60,80,100])
ax[1][1].tick_params(axis="x", labelsize=15)
ax[1][1].tick_params(axis="y", labelsize=15)

#HISTS
ax[0][0].set_xlim(0,7)
for i in range(len(lengths)):
    ax[0][0].hist(lengths[i],bins=10,alpha=0.33,density=True,label=labels)
ax[0][0].set_xlabel("Length ($\\mu m$)",fontsize=20,fontweight="bold")
ax[0][0].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][0].tick_params(axis="x", labelsize=15)
ax[0][0].tick_params(axis="y", labelsize=15)
for i in range(len(widths)):
    ax[0][1].hist(widths[i],bins=10,alpha=0.33,density=True,label=labels)
ax[0][1].set_xlabel("Width ($\\mu m$)",fontsize=20,fontweight="bold")
ax[0][1].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][1].tick_params(axis="x", labelsize=15)
ax[0][1].tick_params(axis="y", labelsize=15)


fig.legend(legendItems, labels,loc='upper center', bbox_to_anchor=(0.5, 1.03), fancybox=False, shadow=False, ncol=3,fontsize="xx-large")
fig.tight_layout()
fig.savefig('ThesisGraphs/GrowthLaw/GrowthLawLengthWidth.pdf', bbox_inches='tight')
fig.savefig('ThesisGraphs/GrowthLaw/GrowthLawLengthWidth.png', bbox_inches='tight')
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************




#**************** INtensity SIZE COMPARISON **** ******************************#
plt.close("all")
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(13,9))
fig.subplots_adjust(left=0.07, right=0.93)
ax[0][0].text(0.05, 0.85 , "A", transform=ax[0][0].transAxes, size=30, weight='bold')
ax[0][1].text(0.05, 0.85 , "B", transform=ax[0][1].transAxes, size=30, weight='bold')
ax[1][0].text(0.05, 0.85 , "C", transform=ax[1][0].transAxes, size=30, weight='bold',color="black")
ax[1][1].text(0.05, 0.85 , "D", transform=ax[1][1].transAxes, size=30, weight='bold')

#Top left plot volume hist
ax[0][0].hist(volumes[0],alpha=0.33,density=True)
ax[0][0].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[0][0].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][0].tick_params(axis="x", labelsize=15)
ax[0][0].tick_params(axis="y", labelsize=15)

#Top right plot intensity
ax[0][1].hist(intensities[0]/10000,alpha=0.33,density=True)
ax[0][1].set_xlabel("Intensity (A.U)",fontsize=20,fontweight="bold")
ax[0][1].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][1].tick_params(axis="x", labelsize=15)
ax[0][1].tick_params(axis="y", labelsize=15)
#Bottom left correlation
hb = ax[1][0].hexbin(volumes[0],intensities[0]/10000,cmap="inferno",gridsize=30,mincnt=1)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax[1][0])
cax = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(hb, cax=cax, orientation='vertical')
ax[1][0].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[1][0].set_ylabel("Intensity (A.U)",fontsize=20,fontweight="bold")
ax[1][0].tick_params(axis="x", labelsize=15)
ax[1][0].tick_params(axis="y", labelsize=15)
#Bottom right
from scipy.stats import spearmanr
for i in range(len(volumes)):
    corr, _ = spearmanr(volumes[i],intensities[i])
    ax[1][1].plot(volumes[i],intensities[i]/10000,'o',alpha=0.33,label ="$\\rho = {0:.2f}$".format(corr) )
ax[1][1].legend(fontsize="x-large")
ax[1][1].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[1][1].set_ylabel("Intensity (A.U)",fontsize=20,fontweight="bold")
ax[1][1].tick_params(axis="x", labelsize=15)
ax[1][1].tick_params(axis="y", labelsize=15)

fig.legend(legendItems, labels,loc='upper center', bbox_to_anchor=(0.5, 1.03), fancybox=False, shadow=False, ncol=3,fontsize="xx-large")
fig.tight_layout()
fig.savefig('ThesisGraphs/GrowthLaw/GrowthLawIntensity.pdf', bbox_inches='tight')
fig.savefig('ThesisGraphs/GrowthLaw/GrowthLawIntensity.png', bbox_inches='tight')
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************

#****************** Surface area to volume ratio *******************************
plt.close("all")
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(9,6))
sa = np.asarray(meanSurfaceAreas)/np.asarray(meanVolumes)
for i in range(len(growthRates)):
    ax.scatter(growthRates[i], sa[i] , c=colors[i],s=150,label=labels[i],marker="D")
ax.plot(sorted(growthRates), sorted(sa) , '--',linewidth=3,color="C6",zorder=0)
print(sorted(growthRates))
print(sorted(sa))
ax.legend( fontsize="xx-large",loc="lower right")
ax.set_xlabel("Doubling time $\\tau$ (min)",fontsize=20,fontweight="bold")
ax.set_ylabel("Surface area to volume ratio",fontsize=20,fontweight="bold")
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
ax.set_ylim(2.5,4.2)
ax.set_xlim(20,100)
fig.tight_layout()
fig.savefig('ThesisGraphs/GrowthLaw/GrowthLawSurface.pdf', bbox_inches='tight')
fig.savefig('ThesisGraphs/GrowthLaw/GrowthLawSurface.png', bbox_inches='tight')
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************


print("****************** DONE *********************")
